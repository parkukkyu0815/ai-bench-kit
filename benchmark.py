#!/usr/bin/env python3
"""AI Service Benchmark — .env만 설정하면 딸깍으로 결과 확인."""

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


def make_azure_client_apikey() -> AzureOpenAI:
    """AOAI (openai lib) — API 키 인증."""
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def make_azure_client_ad() -> AzureOpenAI:
    """AOAI (azure lib) — azure-identity 토큰 인증."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


# ── 스트리밍 호출 + 측정 ──────────────────────────────

def call_chat(client, model: str, system: str, user: str, tools: list | None):
    """Chat Completions API — 스트리밍."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(model=model, messages=msgs, stream=True)
    if tools:
        kwargs["tools"] = tools
    t0 = time.perf_counter()
    ttft = None
    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttft or total, total


def call_responses(client, model: str, system: str, user: str, tools: list | None):
    """Responses API — 스트리밍."""
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
    with client.responses.create(**kwargs) as stream:
        for event in stream:
            if ttft is None and getattr(event, "type", "") == "response.output_text.delta":
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttft or total, total


# ── 조합 생성 ─────────────────────────────────────────

ENDPOINT_LABELS = {
    "openai": "OpenAI 직접",
    "aoai_openai": "AOAI (openai lib)",
    "aoai_azure": "AOAI (azure lib)",
}

API_LABELS = {"chat": "Chat Completions", "responses": "Responses API"}
PROMPT_LABELS = {"short": "짧은 프롬프트", "long": "긴 보안 프롬프트"}
TOOL_LABELS = {False: "OFF", True: "ON"}

CODE_INTERPRETER_TOOL_CHAT = [{"type": "function", "function": {"name": "code_interpreter", "description": "Run code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]
CODE_INTERPRETER_TOOL_RESPONSES = [{"type": "code_interpreter"}]


def build_combos():
    """24가지 조합 생성."""
    endpoints = list(ENDPOINT_LABELS.keys())
    apis = list(API_LABELS.keys())
    prompts = list(PROMPT_LABELS.keys())
    use_tools = [False, True]
    return list(product(endpoints, apis, prompts, use_tools))


# ── 메인 벤치마크 ─────────────────────────────────────

def run_benchmark():
    load_dotenv()

    # 환경 체크
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_azure_key = all(os.environ.get(k) for k in [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION",
    ])
    has_azure_ad = all(os.environ.get(k) for k in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION",
    ])

    if not has_openai and not has_azure_key and not has_azure_ad:
        print("❌ .env에 OpenAI 또는 Azure OpenAI 키를 설정해 주세요.")
        print("   .env.example 파일을 참고하세요.")
        sys.exit(1)

    prompts = load_prompts()
    combos = build_combos()

    # 클라이언트 준비
    clients = {}
    models = {}
    if has_openai:
        clients["openai"] = make_openai_client()
        models["openai"] = "gpt-4.1"
    deploy = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    if has_azure_key:
        clients["aoai_openai"] = make_azure_client_apikey()
        models["aoai_openai"] = deploy
    if has_azure_ad:
        try:
            clients["aoai_azure"] = make_azure_client_ad()
            models["aoai_azure"] = deploy
            print("✓ Azure AD 인증 클라이언트 생성 완료 (az login 필요)")
        except Exception as e:
            print(f"⚠ Azure AD 인증 건너뜀 (az login 필요): {e}")

    # 사용 불가능한 엔드포인트 필터링
    combos = [c for c in combos if c[0] in clients]
    total_calls = len(combos) * len(QUESTIONS) * REPEAT
    print(f"🚀 벤치마크 시작 — {len(combos)}개 조합 × {len(QUESTIONS)}문항 × {REPEAT}회 = {total_calls}회 호출")

    # 결과 저장: results[combo_key][q_id] = [(ttft, total), ...]
    results: dict[tuple, dict[str, list]] = {}

    done = 0
    for combo in combos:
        ep, api, prompt_key, use_tool = combo
        client = clients[ep]
        model = models[ep]
        system = prompts[prompt_key]

        # API 방식에 따른 호출 함수 & 도구 선택
        if api == "chat":
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
                    ttft, total = call_fn(client, model, system, q_text, tools)
                    q_measurements.append((ttft, total))
                    print(f"  ✓ {label} — TTFT={ttft:.3f}s  Total={total:.3f}s")
                except Exception as e:
                    print(f"  ✗ {label} — {e}")
            combo_results[q_id] = q_measurements

        results[combo] = combo_results

    # 결과 리포트 생성
    report = generate_report(results, has_openai, has_azure)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"benchmark_{ts}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n📊 결과 저장: {out_path}")


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
        import azure.identity as _azid
        lines.append(f"| azure-identity | {_azid.__version__} |")
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
