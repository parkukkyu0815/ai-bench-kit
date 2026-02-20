#!/usr/bin/env python3
"""AI Service Benchmark — .env만 설정하면 딸깍으로 결과 확인."""

import json
import os
import re
import sys
import time
import platform
import statistics
from datetime import datetime
from pathlib import Path
from itertools import product

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

# ── 설정 ──────────────────────────────────────────────

REPEAT = 10  # 질문당 반복 횟수

QUESTIONS = [
    ("Q1", "단답형", "대한민국의 수도는?"),
    ("Q2", "설명형", "인플레이션이 금리에 미치는 영향을 설명해줘"),
    ("Q3", "분석형", "금융회사에서 AI 도입 시 고려해야 할 리스크 3가지를 분석해줘"),
    ("Q4", "코드", "Python으로 피보나치 수열 함수를 작성해줘"),
    ("Q5", "복합 추론", "연봉 5000만원 직장인의 소득세를 단계별로 계산해줘"),
]

# ── 프롬프트 로드 (prompts.md에서 읽기) ───────────────

def load_prompts() -> dict[str, str]:
    """prompts.md에서 <!-- BEGIN:key --> … <!-- END:key --> 블록을 파싱."""
    md_path = Path(__file__).parent / "prompts.md"
    text = md_path.read_text(encoding="utf-8")
    prompts = {}
    for m in re.finditer(
        r"<!--\s*BEGIN:(\w+)\s*-->\n(.*?)<!--\s*END:\1\s*-->",
        text,
        re.DOTALL,
    ):
        prompts[m.group(1)] = m.group(2).strip()
    return prompts


# ── 클라이언트 팩토리 ─────────────────────────────────

def make_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def make_azure_openai_client() -> AzureOpenAI:
    """AOAI (openai lib) — openai 라이브러리의 AzureOpenAI 클래스."""
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def make_azure_inference_client():
    """AOAI (azure lib) — azure-ai-inference SDK의 ChatCompletionsClient."""
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    deploy = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    api_ver = os.environ["AZURE_OPENAI_API_VERSION"]

    return ChatCompletionsClient(
        endpoint=f"{endpoint}/openai/deployments/{deploy}",
        credential=AzureKeyCredential(os.environ["AZURE_OPENAI_API_KEY"]),
        api_version=api_ver,
    )


# ── 스트리밍 호출 + 측정 ──────────────────────────────

def call_chat(client, model: str, system: str, user: str, tools: list | None):
    """Chat Completions API (openai lib) — 스트리밍."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(model=model, messages=msgs, stream=True)
    if tools:
        kwargs["tools"] = tools
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(chunk.choices[0].delta.content)
    total = time.perf_counter() - t0
    return ttft or total, total, "".join(chunks)


def call_responses(client, model: str, system: str, user: str, tools: list | None):
    """Responses API (openai lib) — 스트리밍."""
    kwargs = dict(
        model=model,
        instructions=system,
        input=user,
        stream=True,
    )
    if tools:
        kwargs["tools"] = tools
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    with client.responses.create(**kwargs) as stream:
        for event in stream:
            if getattr(event, "type", "") == "response.output_text.delta":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(getattr(event, "delta", ""))
    total = time.perf_counter() - t0
    return ttft or total, total, "".join(chunks)


def call_chat_azure_inference(client, _model: str, system: str, user: str, tools: list | None):
    """Chat Completions (azure-ai-inference SDK) — 스트리밍."""
    from azure.ai.inference.models import SystemMessage, UserMessage

    msgs = [SystemMessage(content=system), UserMessage(content=user)]
    kwargs = dict(messages=msgs, stream=True)
    if tools:
        kwargs["tools"] = tools
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    response = client.complete(**kwargs)
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(chunk.choices[0].delta.content)
    total = time.perf_counter() - t0
    return ttft or total, total, "".join(chunks)


# ── 조합 생성 ─────────────────────────────────────────

ENDPOINT_LABELS = {
    "openai": "OpenAI 직접",
    "aoai_openai": "AOAI (openai lib)",
    "aoai_azure": "AOAI (azure lib)",
}

API_LABELS = {"chat": "Chat Completions", "responses": "Responses API"}
PROMPT_LABELS = {"short": "짧은 프롬프트", "long": "긴 보안 프롬프트"}
TOOL_LABELS = {False: "OFF", True: "ON"}

# azure-ai-inference는 Chat Completions만 지원
ENDPOINT_SUPPORTED_APIS = {
    "openai": {"chat", "responses"},
    "aoai_openai": {"chat", "responses"},
    "aoai_azure": {"chat"},
}

CODE_INTERPRETER_TOOL_CHAT = [{"type": "function", "function": {"name": "code_interpreter", "description": "Run code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]
CODE_INTERPRETER_TOOL_RESPONSES = [{"type": "code_interpreter"}]


def build_combos():
    """조합 생성 (엔드포인트별 지원 API 필터링)."""
    endpoints = list(ENDPOINT_LABELS.keys())
    apis = list(API_LABELS.keys())
    prompts = list(PROMPT_LABELS.keys())
    use_tools = [False, True]
    combos = []
    for ep, api, prompt_key, use_tool in product(endpoints, apis, prompts, use_tools):
        if api in ENDPOINT_SUPPORTED_APIS.get(ep, set()):
            combos.append((ep, api, prompt_key, use_tool))
    return combos


# ── 메인 벤치마크 ─────────────────────────────────────

def run_benchmark():
    load_dotenv()

    # 환경 체크
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_azure = all(os.environ.get(k) for k in [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION",
    ])

    if not has_openai and not has_azure:
        print("❌ .env에 OpenAI 또는 Azure OpenAI 키를 설정해 주세요.")
        print("   .env.example 파일을 참고하세요.")
        sys.exit(1)

    prompts = load_prompts()

    # 클라이언트 준비
    clients = {}
    models = {}
    if has_openai:
        clients["openai"] = make_openai_client()
        models["openai"] = "gpt-5-chat"
    if has_azure:
        deploy = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        clients["aoai_openai"] = make_azure_openai_client()
        models["aoai_openai"] = deploy
        clients["aoai_azure"] = make_azure_inference_client()
        models["aoai_azure"] = deploy

    # 사용 가능한 조합만 필터링
    combos = [c for c in build_combos() if c[0] in clients]
    total_calls = len(combos) * len(QUESTIONS) * REPEAT

    print(f"🚀 벤치마크 시작 — {len(combos)}개 조합 × {len(QUESTIONS)}문항 × {REPEAT}회 = {total_calls}회 호출")
    if has_azure:
        print(f"   ℹ AOAI (azure lib)은 Chat Completions만 지원 → Responses API 제외")

    # 결과 저장: results[combo_key][q_id] = [(ttft, total), ...]
    results: dict[tuple, dict[str, list]] = {}

    done = 0
    for combo in combos:
        ep, api, prompt_key, use_tool = combo
        client = clients[ep]
        model = models[ep]
        system = prompts[prompt_key]

        # 엔드포인트 + API 방식에 따른 호출 함수 & 도구 선택
        if ep == "aoai_azure":
            call_fn = call_chat_azure_inference
            tools = CODE_INTERPRETER_TOOL_CHAT if use_tool else None
        elif api == "chat":
            call_fn = call_chat
            tools = CODE_INTERPRETER_TOOL_CHAT if use_tool else None
        else:
            call_fn = call_responses
            tools = CODE_INTERPRETER_TOOL_RESPONSES if use_tool else None

        combo_results: dict[str, list] = {}

        for q_id, q_diff, q_text in QUESTIONS:
            q_measurements = []
            for r in range(REPEAT):
                done += 1
                label = f"[{done}/{total_calls}] {ENDPOINT_LABELS[ep]} | {API_LABELS[api]} | {PROMPT_LABELS[prompt_key]} | 도구={TOOL_LABELS[use_tool]} | {q_id} #{r+1}"
                try:
                    ttft, total, text = call_fn(client, model, system, q_text, tools)
                    q_measurements.append((ttft, total, text))
                    print(f"  ✓ {label} — TTFT={ttft:.3f}s  Total={total:.3f}s")
                except Exception as e:
                    print(f"  ✗ {label} — {e}")
            combo_results[q_id] = q_measurements

        results[combo] = combo_results

    # 결과 저장
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 기존 통계 리포트
    report = generate_report(results, has_openai, has_azure)
    out_path = results_dir / f"benchmark_{ts}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n📊 통계 리포트: {out_path}")

    # 2) raw 결과 JSON + 질문별 비교 마크다운
    raw_path = save_raw_results(results, models, results_dir, ts)
    print(f"📦 Raw 결과: {raw_path}")
    compare_path = save_comparison(results, models, results_dir, ts)
    print(f"🔍 비교용 파일: {compare_path}")


# ── 통계 헬퍼 ─────────────────────────────────────────

def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
    }


def fmt(v: float) -> str:
    return f"{v:.3f}"


def combo_label(combo: tuple) -> str:
    ep, api, prompt_key, use_tool = combo
    return f"{ENDPOINT_LABELS[ep]} | {API_LABELS[api]} | {PROMPT_LABELS[prompt_key]} | 도구={TOOL_LABELS[use_tool]}"


def save_raw_results(results: dict, models: dict, results_dir: Path, ts: str) -> Path:
    """전체 raw 결과를 구조화된 JSON으로 저장."""
    q_map = {q[0]: {"difficulty": q[1], "text": q[2]} for q in QUESTIONS}
    records = []
    for combo, q_results in results.items():
        ep, api, prompt_key, use_tool = combo
        for q_id, measurements in q_results.items():
            for i, (ttft, total, text) in enumerate(measurements):
                records.append({
                    "endpoint": ep,
                    "endpoint_label": ENDPOINT_LABELS[ep],
                    "model": models.get(ep, ""),
                    "api": api,
                    "api_label": API_LABELS[api],
                    "prompt": prompt_key,
                    "prompt_label": PROMPT_LABELS[prompt_key],
                    "tool": use_tool,
                    "question_id": q_id,
                    "question_difficulty": q_map[q_id]["difficulty"],
                    "question_text": q_map[q_id]["text"],
                    "repeat": i + 1,
                    "ttft": round(ttft, 4),
                    "total": round(total, 4),
                    "response": text,
                })
    out = results_dir / f"raw_{ts}.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def save_comparison(results: dict, models: dict, results_dir: Path, ts: str) -> Path:
    """질문별로 모든 조합의 응답을 나란히 보여주는 비교용 마크다운 생성."""
    lines = ["# 질문별 응답 비교", ""]
    lines.append(f"> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("같은 질문에 대한 각 조합의 응답을 나란히 비교할 수 있습니다.")
    lines.append("다른 AI 모델에게 이 파일을 전달하여 품질 평가를 요청할 수 있습니다.")
    lines.append("")

    for q_id, q_diff, q_text in QUESTIONS:
        lines.append(f"---")
        lines.append(f"## {q_id} ({q_diff})")
        lines.append(f"")
        lines.append(f"**질문:** {q_text}")
        lines.append("")

        for combo, q_results in results.items():
            if q_id not in q_results or not q_results[q_id]:
                continue
            ep = combo[0]
            tag = combo_label(combo)
            measurements = q_results[q_id]

            lines.append(f"### {tag}")
            lines.append(f"- 모델: `{models.get(ep, '')}`")
            lines.append("")

            for i, (ttft, total, text) in enumerate(measurements):
                lines.append(f"<details><summary>반복 {i+1} — TTFT {ttft:.3f}s, Total {total:.3f}s</summary>")
                lines.append("")
                lines.append(text)
                lines.append("")
                lines.append("</details>")
                lines.append("")

    out = results_dir / f"compare_{ts}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def stats_row(label: str, values: list[float]) -> str:
    s = stats(values)
    return f"| {label} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['min'])} | {fmt(s['max'])} | {fmt(s['std'])} |"


# ── 리포트 생성 ───────────────────────────────────────

def collect_env_info(has_openai: bool, has_azure: bool) -> str:
    lines = []
    lines.append(f"| OS | {platform.system()} {platform.release()} |")
    lines.append(f"| Python | {platform.python_version()} |")
    try:
        import openai as _openai
        lines.append(f"| openai 라이브러리 | {_openai.__version__} |")
    except Exception:
        pass
    try:
        from azure.ai.inference import __version__ as _azver
        lines.append(f"| azure-ai-inference | {_azver} |")
    except Exception:
        pass
    lines.append(f"| 테스트 시각 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    if has_azure:
        ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        m = re.search(r"https://[^.]*\.(\w+)\.api", ep) or re.search(r"https://(\w+)\.", ep)
        region = m.group(1) if m else "알 수 없음"
        lines.append(f"| Azure 리전 | {region} |")
    return "\n".join(lines)


def generate_report(results: dict, has_openai: bool, has_azure: bool) -> str:
    """변수별 그룹핑 분석 리포트 생성."""
    lines = [f"# AI Service Benchmark 결과", ""]
    lines.append(f"> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 환경 정보
    lines.append("## 테스트 환경")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|------|---|")
    lines.append(collect_env_info(has_openai, has_azure))
    lines.append("")

    # 전체 결과를 평탄화: (combo, q_id) → ttft/total 리스트
    flat_ttft: dict[tuple, list[float]] = {}
    flat_total: dict[tuple, list[float]] = {}
    for combo, q_results in results.items():
        for q_id, measurements in q_results.items():
            flat_ttft[(combo, q_id)] = [m[0] for m in measurements]
            flat_total[(combo, q_id)] = [m[1] for m in measurements]

    def group_values(group_fn):
        """group_fn(combo) → group_key 로 그룹핑하여 ttft/total 통합."""
        groups_ttft: dict[str, list[float]] = {}
        groups_total: dict[str, list[float]] = {}
        for (combo, q_id), vals in flat_ttft.items():
            key = group_fn(combo)
            groups_ttft.setdefault(key, []).extend(vals)
            groups_total.setdefault(key, []).extend(flat_total[(combo, q_id)])
        return groups_ttft, groups_total

    def write_table(title: str, group_fn, label_map: dict):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("### TTFT (Time To First Token, 초)")
        lines.append("")
        lines.append("| 구분 | 평균 | 중앙값 | 최솟값 | 최댓값 | 표준편차 |")
        lines.append("|------|------|--------|--------|--------|----------|")
        g_ttft, g_total = group_values(group_fn)
        for key in label_map:
            if key in g_ttft:
                lines.append(stats_row(label_map[key], g_ttft[key]))
        lines.append("")
        lines.append("### Total Latency (초)")
        lines.append("")
        lines.append("| 구분 | 평균 | 중앙값 | 최솟값 | 최댓값 | 표준편차 |")
        lines.append("|------|------|--------|--------|--------|----------|")
        for key in label_map:
            if key in g_total:
                lines.append(stats_row(label_map[key], g_total[key]))
        lines.append("")

    # 1. 엔드포인트 비교
    write_table(
        "1. 엔드포인트 비교",
        lambda c: c[0],
        ENDPOINT_LABELS,
    )

    # 2. API 방식 비교
    write_table(
        "2. API 방식 비교",
        lambda c: c[1],
        API_LABELS,
    )

    # 3. 시스템 프롬프트 영향
    write_table(
        "3. 시스템 프롬프트 영향",
        lambda c: c[2],
        PROMPT_LABELS,
    )

    # 4. 코드 인터프리터 영향
    write_table(
        "4. 코드 인터프리터 영향",
        lambda c: c[3],
        TOOL_LABELS,
    )

    # 5. 질문 난이도별 비교
    q_labels = {q[0]: f"{q[0]} ({q[1]})" for q in QUESTIONS}
    lines.append("## 5. 질문 난이도별 비교")
    lines.append("")
    lines.append("### TTFT (Time To First Token, 초)")
    lines.append("")
    lines.append("| 구분 | 평균 | 중앙값 | 최솟값 | 최댓값 | 표준편차 |")
    lines.append("|------|------|--------|--------|--------|----------|")
    q_ttft: dict[str, list[float]] = {}
    q_total: dict[str, list[float]] = {}
    for (combo, q_id), vals in flat_ttft.items():
        q_ttft.setdefault(q_id, []).extend(vals)
        q_total.setdefault(q_id, []).extend(flat_total[(combo, q_id)])
    for q_id in q_labels:
        if q_id in q_ttft:
            lines.append(stats_row(q_labels[q_id], q_ttft[q_id]))
    lines.append("")
    lines.append("### Total Latency (초)")
    lines.append("")
    lines.append("| 구분 | 평균 | 중앙값 | 최솟값 | 최댓값 | 표준편차 |")
    lines.append("|------|------|--------|--------|--------|----------|")
    for q_id in q_labels:
        if q_id in q_total:
            lines.append(stats_row(q_labels[q_id], q_total[q_id]))
    lines.append("")

    # 6. 전체 조합 상세
    lines.append("## 6. 전체 조합 상세")
    lines.append("")
    lines.append("| 엔드포인트 | API | 프롬프트 | 도구 | 문항 | TTFT 평균 | Total 평균 |")
    lines.append("|-----------|-----|---------|------|------|-----------|-----------|")
    for combo, q_results in results.items():
        ep, api, prompt_key, use_tool = combo
        for q_id, measurements in q_results.items():
            if not measurements:
                continue
            ttft_vals = [m[0] for m in measurements]
            total_vals = [m[1] for m in measurements]
            lines.append(
                f"| {ENDPOINT_LABELS[ep]} | {API_LABELS[api]} | "
                f"{PROMPT_LABELS[prompt_key]} | {TOOL_LABELS[use_tool]} | "
                f"{q_id} | {fmt(statistics.mean(ttft_vals))} | {fmt(statistics.mean(total_vals))} |"
            )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run_benchmark()
