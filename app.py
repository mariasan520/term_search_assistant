import ast
import json
import re
import time
from dataclasses import dataclass

import certifi
import pandas as pd
import requests
import streamlit as st
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 🔴 强行关闭控制台的 SSL 禁用警告，让后台日志保持干净
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 页面基本配置与全局常量映射
# ==========================================
st.set_page_config(
    page_title="术语检索助手",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 核心 API 服务商参数映射字典（🛠️ 彻底修复：严格对齐 Google 官方标准 OpenAI 兼容路由，移除多余的 /v1，根除 404）
PROVIDER_MAP = {
    "硅基流动 (SiliconFlow)": {
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "Qwen/Qwen2.5-72B-Instruct",
    },
    "DeepSeek 官方": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
    },
    # ✅ Gemini 使用原生接口
    "Google Gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "model": "gemini-2.5-flash",
    },
    "自定义 OpenAI 兼容接口": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4-turbo",
    },
}

MODEL_ROUTE = {
    "Google Gemini": {
        "simple": "gemini-2.5-flash",
        "complex": "gemini-2.5-pro",
    }
}

REQUEST_TIMEOUT = 40

# 🌐 默认数据源
DEFAULT_SOURCES = {
    "标准组织": ["ITU-T", "3GPP", "ETSI", "IETF RFC", "IEEE", "ISO", "IEC", "NIST"],
    "厂商": ["Cisco", "Juniper", "Nokia", "Ericsson", "VMware", "Red Hat", "Huawei"],
    "运营商": ["Verizon", "AT&T"],
    "开源社区与论坛": [
        "Linux Kernel",
        "CNCF",
        "Kubernetes Community",
        "OpenConfig",
        "IETF Datatracker",
        "Network to Code Community",
        "Palo Alto LIVEcommunity",
        "NANOG",
        "RIPE Community",
    ],
}

# 注入优化后的 CSS 样式
st.markdown(
    """
    <style>
    div[data-testid="stToolbar"] { visibility: hidden; }
    .main .block-container { padding-top: 2rem !important; }
    div[data-testid="column"]:has(button[key="btn_top_settings"]) { display: flex; justify-content: flex-end; align-items: flex-start; }
    
    /* 为原生 HTML 表格注入定制化样式，替代原有 st.table 表现 */
    .custom-rendered-table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9rem; }
    .custom-rendered-table th { background-color: #f8fafc; color: #1e293b; font-weight: 600; padding: 12px; text-align: left; border: 1px solid #e2e8f0; white-space: nowrap !important; }
    .custom-rendered-table td { padding: 12px; border: 1px solid #e2e8f0; color: #334155; vertical-align: top; white-space: normal !important; word-break: break-word !important; }
    .custom-rendered-table tr:nth-child(even) { background-color: #f8fafc; }
    .custom-rendered-table a { color: #2563eb !important; text-decoration: underline !important; font-weight: 500; }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. 全局状态存储器初始化
# ==========================================
@dataclass
class SearchConfig:
    search_active: bool = False
    has_searched: bool = False
    latest_result: list = None


defaults = {
    "saved_api_key": "",
    "saved_provider": "硅基流动 (SiliconFlow)",
    "last_provider": "硅基流动 (SiliconFlow)",
    "custom_provider_name": "",
    "search_active": False,
    "has_searched": False,
    "latest_result": None,
    "is_verification_mode": False,
    "last_search_zh": "",
    "last_search_en": "",
    "active_sources": [
        "ITU-T",
        "3GPP",
        "ETSI",
        "IETF RFC",
        "IEEE",
        "ISO",
        "IEC",
        "Cisco",
        "Juniper",
        "Nokia",
        "Huawei",
    ],
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

form_keys = [
    "form_zh",
    "form_en",
    "form_domain",
    "form_tech",
    "form_keyword",
    "form_context",
]
for k in form_keys:
    if k not in st.session_state:
        st.session_state[k] = ""


# ==========================================
# 3. 核心大模型网络请求封装函数
# ==========================================
def build_session():
    s = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    return s


HTTP = requests.Session()


def fetch_ai_completion(provider, api_key, system_prompt, user_prompt):
    config = PROVIDER_MAP.get(provider, PROVIDER_MAP["自定义 OpenAI 兼容接口"])
    url = config["url"]
    model = config["model"]

    if provider == "Google Gemini":

        total_len = (
            len(system_prompt)
        +
            len(user_prompt)
    )

        model = (
            MODEL_ROUTE[
                provider
            ]["simple"]

            if total_len < 6000

            else

            MODEL_ROUTE[
                provider
            ]["complex"]
    )

    try:
        # ======================
        # Gemini 专用
        # ======================
        if provider == "Google Gemini":
            headers = {"Content-Type": "application/json"}
            payload = {
                "generationConfig": {"temperature": 0.1},
                "contents": [
                    {"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                ],
            }

            # Gemini 单独证书策略
            ssl_verify = certifi.where()

            try:

                res = build_session().post(
                    url,
                    headers=headers,
                    params={
                        "key": api_key
                    },
                    json=payload,
                    timeout=40,
                    verify=ssl_verify
                )

            except requests.exceptions.SSLError:

                print(
                    "⚠️ Gemini SSL失败，自动降级"
                )

                res = build_session().post(
                    url,
                    headers=headers,
                    params={
                        "key": api_key
                    },
                    json=payload,
                    timeout=40,
                    verify=False
                )

            print("Gemini STATUS:", res.status_code)
            print("Gemini BODY:", res.text)
            res.raise_for_status()
            data = res.json()
            raw_content = data["candidates"][0]["content"]["parts"][0]["text"]

        # ======================
        # OpenAI 兼容
        # ======================
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            }

            res = build_session().post(
                url,
                headers=headers,
                json=payload,
                timeout=40,
                verify=certifi.where(),
            )

            print("STATUS:", res.status_code)
            print("BODY:", res.text)
            res.raise_for_status()
            data = res.json()

            if "choices" not in data or not data["choices"]:
                raise Exception("未发现 choices 返回")

            raw_content = data["choices"][0]["message"]["content"]

        clean_content = re.sub(
            r"^```json\s*|```$", "", raw_content.strip(), flags=re.IGNORECASE
        ).strip()
        match = re.search(r"(\[.*\]|\{.*\})", clean_content, re.DOTALL)
        if match:
            clean_content = match.group(1)

        try:
            return json.loads(clean_content)
        except Exception:
            try:
                return ast.literal_eval(clean_content)
            except Exception:
                repaired = clean_content.replace("\n", " ").replace("'", '"')
                try:
                    return json.loads(repaired)
                except Exception:
                    raise Exception("模型未输出合法JSON")

    except requests.exceptions.HTTPError as e:
        print("HTTP ERROR")
        print(res.text)
        raise Exception(f"接口异常：{res.status_code}\n{res.text}")
    except Exception as e:
        raise Exception(f"解析失败：{str(e)}")


# ==========================================
# 4. 定义“设置”对话框
# ==========================================
@st.dialog("🛠️ 设置")
def show_settings_dialog():
    current_provider = st.session_state["saved_provider"]
    options_list = list(PROVIDER_MAP.keys())
    default_index = (
        options_list.index(current_provider)
        if current_provider in options_list
        else options_list.index("自定义 OpenAI 兼容接口")
    )

    api_provider_input = st.selectbox(
        "API 服务商通道", options_list, index=default_index
    )

    if api_provider_input != st.session_state["last_provider"]:
        st.session_state["saved_api_key"] = ""
        st.session_state["last_provider"] = api_provider_input

    final_provider_name = api_provider_input
    if api_provider_input == "自定义 OpenAI 兼容接口":
        custom_name_input = st.text_input(
            "请输入自定义服务商通道名称",
            placeholder="例如：我的本地大模型 / 公司内部网关...",
            value=st.session_state["custom_provider_name"],
        )
        if custom_name_input.strip():
            final_provider_name = custom_name_input.strip()

    api_key_input = st.text_input(
        "API 密钥 (API Key)",
        type="password",
        placeholder="请在此粘贴您的 API Key...",
        value=st.session_state["saved_api_key"],
    )
    st.markdown("--- \n ##### 🌐 默认数据源配置")

    selected_sources = []
    for category, sources in DEFAULT_SOURCES.items():
        with st.expander(category):
            for src in sources:
                state_key = f"src_{category}_{src}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = (
                        src in st.session_state["active_sources"]
                    )
                if st.checkbox(
                    src,
                    value=st.session_state[state_key],
                    key=f"dialog_{state_key}",
                ):
                    selected_sources.append(src)

    custom_sources = st.text_input(
        "其它自定义数据源", placeholder="多个用逗号隔开", key="dialog_custom_sources"
    )
    if custom_sources:
        selected_sources.extend(
            [s.strip() for s in custom_sources.split(",") if s.strip()]
        )

    if st.button("💾 保存", use_container_width=True):
        st.session_state.update(
            {
                "saved_api_key": api_key_input,
                "saved_provider": final_provider_name,
                "custom_provider_name": (
                    final_provider_name
                    if api_provider_input == "自定义 OpenAI 兼容接口"
                    else ""
                ),
                "active_sources": selected_sources,
            }
        )
        st.success("✅ 配置已锁定！设置成功。")
        time.sleep(0.6)
        st.rerun()


# ==========================================
# 5. 顶栏头部与布局划分
# ==========================================
hl, hr = st.columns([8.5, 1.5])
hl.markdown(
    """
    <div style="padding-bottom: 10px; border-bottom: 1px solid #e6e9ef; margin-bottom: 25px;">
        <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700; color: #1e293b;">🔍 术语检索助手</h1>
        <p style="margin: 8px 0 0 0; padding: 0; color: #64748b; font-size: 0.9rem;">💡 请在下方输入单个中文术语。若输入英文则自动启动【术语查证】模式，留空则自动运行【新术语检索】。</p>
    </div>
""",
    unsafe_allow_html=True,
)

with hr:
    st.markdown(
        "<div style='margin-top: 5px; text-align: right;'>",
        unsafe_allow_html=True,
    )
    if st.button("⚙️ 设置", key="btn_top_settings"):
        show_settings_dialog()
    st.markdown("</div>", unsafe_allow_html=True)

layout_main, layout_space = st.columns([2, 1])

with layout_main:
    input_suffix = "_active" if st.session_state.get("clear_triggered") else ""

    left_col, right_col = st.columns(2)
    with left_col:
        text_zh_val = st.session_state.get("form_zh", "")
        text_en_val = st.session_state.get("form_en", "")
        text_dom_val = st.session_state.get("form_domain", "")
        zh_term = st.text_input(
            "中文术语 *",
            value=text_zh_val,
            placeholder="输入需要检索的中文术语。",
            key=f"input_zh{input_suffix}",
        )
        en_term = st.text_input(
            "英文术语 (可选)",
            value=text_en_val,
            placeholder="输入需要查证的英文表达。",
            key=f"input_en{input_suffix}",
        )
        domain = st.text_input(
            "使用领域 (可选)",
            value=text_dom_val,
            placeholder="输入术语的使用领域。",
            key=f"input_domain{input_suffix}",
        )
    with right_col:
        text_tech_val = st.session_state.get("form_tech", "")
        text_kw_val = st.session_state.get("form_keyword", "")
        text_ctx_val = st.session_state.get("form_context", "")
        tech_point = st.text_input(
            "技术点 (可选)",
            value=text_tech_val,
            placeholder="输入术语相关的技术点。",
            key=f"input_tech{input_suffix}",
        )
        keyword = st.text_input(
            "关键词 (可选)",
            value=text_kw_val,
            placeholder="输入与术语经常一起使用或相关联的词语。多个关键词用逗号隔开。",
            key=f"input_kw{input_suffix}",
        )
        context = st.text_area(
            "上下文语境 (可选)",
            value=text_ctx_val,
            placeholder="输入包含该术语的上下文描述...",
            height=68,
            key=f"input_ctx{input_suffix}",
        )

    st.session_state["form_zh"] = zh_term
    st.session_state["form_en"] = en_term
    st.session_state["form_domain"] = domain
    st.session_state["form_tech"] = tech_point
    st.session_state["form_keyword"] = keyword
    st.session_state["form_context"] = context

    # 保持清除、取消、按钮顺序完全不变
    btn_col1, btn_col2, btn_col3, btn_space = st.columns([1.5, 1.5, 1.5, 5.5])
    search_btn = btn_col1.button(
        "🚀 检索", key="btn_start_run", use_container_width=True
    )
    cancel_btn = btn_col2.button(
        "❌ 取消", key="btn_cancel_run", use_container_width=True
    )
    clear_btn = btn_col3.button(
        "🧹 清空", key="btn_clear_fields", use_container_width=True
    )

if clear_btn:
    current_latest_result = st.session_state.get("latest_result", None)
    current_is_verify = st.session_state.get("is_verification_mode", False)
    current_last_zh = st.session_state.get("last_search_zh", "")
    current_last_en = st.session_state.get("last_search_en", "")

    for k in form_keys:
        st.session_state[k] = ""

    st.session_state["clear_triggered"] = not st.session_state.get(
        "clear_triggered", False
    )

    st.session_state["latest_result"] = current_latest_result
    st.session_state["is_verification_mode"] = current_is_verify
    st.session_state["last_search_zh"] = current_last_zh
    st.session_state["last_search_en"] = current_last_en
    st.session_state["has_searched"] = (
        True if current_latest_result else False
    )

    st.rerun()

if cancel_btn:
    if st.session_state["search_active"]:
        st.session_state.update(
            {
                "search_active": False,
                "has_searched": False,
                "latest_result": None,
                "is_verification_mode": False,
                "last_search_zh": "",
                "last_search_en": "",
            }
        )
        st.warning("🛑 任务已被手动取消。")
        time.sleep(0.5)
        st.rerun()
    else:
        st.info("ℹ️ 当前没有正在运行的任务。")

# ==========================================
# 6. 核心业务分流与联网请求逻辑
# ==========================================
if search_btn:
    st.session_state["search_active"] = True
    cleaned_zh, cleaned_en = zh_term.strip(), en_term.strip()

    if not cleaned_zh:
        st.error("❌ 错误：‘中文术语’为必填项，请输入后再点击检索。")
        st.session_state["search_active"] = False
    elif any(char in cleaned_zh for char in ["，", ",", "、", " ", "\t"]):
        st.error("❌ 错误：‘中文术语’一栏目前仅支持输入【1个】指定术语。")
        st.session_state["search_active"] = False
    elif not st.session_state["saved_api_key"]:
        st.error(
            "❌ 错误：未检测到有效的 API 密钥。请点击右上角 [⚙️ 设置] 配置。"
        )
        st.session_state["search_active"] = False
    else:
        sources_string = ", ".join(st.session_state["active_sources"])
        current_mode_is_verify = bool(cleaned_en)

        st.session_state["last_search_zh"] = cleaned_zh
        st.session_state["last_search_en"] = (
            cleaned_en if current_mode_is_verify else "--"
        )

        if current_mode_is_verify:
            mode_instruction = f"""【💡 当前核心任务：术语合规性查证模式】
用户输入了需要验证的英文表达："{cleaned_en}"。你必须在指定数据源（包含：{sources_string}）中检索查证。
特别注意：如果该术语在行业中跨越了多个不同的子领域（例如云计算、无线通信、数通网络等），你必须在返回的 JSON 列表中针对每一个领域分别创建一行独立对象，在各子领域下独立判断该英文表达是否属于推荐译法。"""
        else:
            mode_instruction = f"""【💡 当前核心任务：新术语检索模式】
用户未输入任何英文表达。你必须根据中文术语，在指定数据源（包含：{sources_string}）中进行开放式多维全网检索。
特别注意：如果该中文术语跨多个应用或技术领域，你必须将其拆分为多条记录，按领域分别输出。"""

        system_prompt = f"""你是一名在ICT及数据通信（Datacom）领域从业 20 年的资盛专家级翻译工程与技术术语审计专家。
当前用户给出了一个中文术语，需要你结合选定的数据源进行全网多维文献检索与行业辨析。

{mode_instruction}

【🔴 绝对禁止输出任何原生 HTML 表格标签（如 <tr>, <td>, <table>）】：
你必须完全且仅仅以合规的标准 JSON 数组（List of Objects）结构输出数据。

【🔴 译法数量与定量供应铁律】：
针对“推荐译法”、“其它可接受译法”和“不推荐译法”这三列，在任何场景下均必须提供 1 至 5 个行业英文参考表达，其中以提供 3 至 5 个专业英文译法为最佳（若有多项请用英文逗号 `, ` 分隔）。

【🔴 一词多领域拆分规则】：
如果检索到该术语适用于多个不同领域，必须在 JSON 数组中按领域拆分成多个独立对象。例如有2个领域，则输出含有2个对象的数组。

【🔴 关键词关联检索专项指令】：
当用户提供了“关联关键词”时，你在检索和筛选英文术语时，必须优先检索并推荐那些在标准文献、厂商官方文档中与该指定关键词经常一起搭配使用或有着强关联的英文专业术语表达。请在“备注”中扼要说明该术语与关键词的关联用法。

【🔴 数据来源链接生成专项指令】：
对于选定的数据源（包含：{sources_string}），你必须在 JSON 中提供一个名为 "数据来源与链接" 的属性。在这个属性中，请列出你推导、查证该词所参考的真实权威数据源名称（例如 3GPP Specs、Cisco Support Docs、IETF RFC 等）。
极其重要：请尽量将数据源包装成标准 HTML 链接格式，例如 `<a href="https://tools.ietf.org/html/rfc..." target="_blank">IETF RFC</a>`。如果无法提供特定文章的长尾完整 URL，请直接使用该组织/厂商术语库的官方主站入口 URL（例如：`<a href="https://www.3gpp.org" target="_blank">3GPP Specification Center</a>`）。绝对保证链接可用！

【输出格式要求】：
你必须严格输出一个规整的 JSON 数组（List of Objects），格式如下 : 
[
  {{"是否推荐": "例如：是 / 否 / 行业推荐","使用领域": "...","权威定义": "...","推荐译法": "...","双语示例": "...","其它可接受译法": "...","不推荐译法": "...","备注": "...","数据来源与链接": "..."}}
]"""

        user_prompt = f"中文术语: {cleaned_zh}\n输入的英文术语(可选): {cleaned_en}\n用户指定的使用领域(可选): {domain.strip()}\n技术点(可选): {tech_point.strip()}\n关联关键词(可选): {keyword.strip()}\n上下文语境(可选): {context.strip()}"

        loading = st.empty()

        try:

            with loading.container():

                with st.spinner(
                   f"🌐 正在跨源多领域检索并审计【{cleaned_zh}】..."
                ):

                    result_list = [{"使用领域": "test", "推荐译法": "test"}]

                    if isinstance(result_list, dict):
                        result_list = [result_list]

                    key_mapping = {
                        "是否推荐": ["是否推荐", "推荐级别"],
                        "使用领域": ["使用领域", "领域"],
                        "权威定义": ["权威定义", "定义"],
                        "推荐译法": ["推荐译法", "建议译法"],
                        "双语示例": ["双语示例", "示例"],
                        "其它可接受译法": [
                            "其它可接受译法",
                            "可接受译法"
                        ],
                        "不推荐译法": [
                            "不推荐译法",
                            "禁用译法"
                        ],
                        "备注": [
                            "备注",
                            "说明"
                        ],
                        "数据来源与链接": [
                            "数据来源与链接",
                            "数据来源",
                            "来源链接",
                            "来源",
                        ],
                    }

                    normalized_list = []

                    for item in result_list:

                        norm_item = {

                            target_key:

                            next(
                                (
                                    item[a]

                                    for a in aliases

                                    if a in item
                                ),

                                "--"

                            )

                            for target_key, aliases

                            in key_mapping.items()

                        }

                        normalized_list.append(
                            norm_item
                        )

                    st.session_state.update(
                        {
                            "has_searched": True,
                            "latest_result": normalized_list,
                            "is_verification_mode": current_mode_is_verify,
                            "search_active": False,
                        }
                    )

                    st.rerun()

        except requests.exceptions.Timeout:

            st.error(
                "❌ 请求超时，请稍后重试。"
            )

        except Exception as e:

            st.error(
                f"❌ 解析异常或凭证拒绝，已重置防线，请重新检索。详情: {e}"
            )

        finally:

            loading.empty()

            st.session_state[
                "search_active"
            ] = False

# ==========================================
# 7. 结果展现渲染区
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)

if st.session_state.get("latest_result"):
    st.session_state["has_searched"] = True

if st.session_state["has_searched"] and st.session_state["latest_result"]:
    st.markdown("### 📊 检索结果")
    res_list = st.session_state["latest_result"]

    def clean_val(val):
        if val is None:
            return "--"
        text = str(val).strip()
        text = re.sub(
            r"\(查不到写[\'\"]?--[\'\"]?\)|\(如果没输英文.*?\)|\(若无则填.*?\)",
            "",
            text,
        )
        return text if text else "--"

    def clean_english_only(val):
        text = clean_val(val)
        if text == "--":
            return "--"
        clean_text = re.sub(r"[\u4e00-\u9fa5]+", "", text)
        clean_text = re.sub(r"\(\s*\)|（\s*）", "", clean_text)
        clean_text = re.sub(
            r"^[,\s;，；、/\\\.。]+|[,\s;，；、/\\\.。]+$", "", clean_text
        )
        return (
            clean_text.strip()
            if clean_text.strip() not in ["", "none", "None", "null"]
            else "--"
        )

    zh_display = st.session_state.get("last_search_zh", "--")
    en_display = st.session_state.get("last_search_en", "--")

    merged_records = {}
    is_verify_mode = st.session_state.get("is_verification_mode", False)
    target_en_lower = en_display.lower()

    for res in res_list:
        domain_val = clean_val(
            res.get("使用领域") or st.session_state["form_domain"] or "--"
        )
        is_rec = clean_val(res.get("是否推荐", "--"))
        rec_trans = clean_english_only(res.get("推荐译法", "--"))
        auth_def = clean_val(res.get("权威定义", "--"))
        bilingual_eg = clean_val(res.get("双语示例", "--"))
        other_trans = clean_english_only(res.get("其它可接受译法", "--"))
        no_rec_trans = clean_english_only(res.get("不推荐译法", "--"))
        remark_val = clean_val(res.get("备注", "--"))
        source_link = str(res.get("数据来源与链接", "--")).strip()

        if is_verify_mode and is_rec in ["是", "行业推荐", "官方推荐"]:
            if no_rec_trans != "--":
                terms = [
                    t.strip() for t in no_rec_trans.split(",") if t.strip()
                ]
                filtered_terms = [
                    t for t in terms if t.lower() != target_en_lower
                ]
                no_rec_trans = (
                    ", ".join(filtered_terms) if filtered_terms else "--"
                )
            if any(
                neg in remark_val
                for neg in ["不推荐使用", "错误术语", "不建议用", "并非标准"]
            ):
                remark_val = f"【查证合规】经权威编盛，该英文术语在当前领域符合标准。原备注修正：{remark_val}"

        if domain_val not in merged_records:
            merged_records[domain_val] = {
                "是否推荐": [is_rec],
                "推荐译法": [rec_trans],
                "权威定义": [auth_def],
                "双语示例": [bilingual_eg],
                "其它可接受译法": [other_trans],
                "不推荐译法": [no_rec_trans],
                "备注说明": [remark_val],
                "数据来源与链接": [source_link],
            }
        else:
            current_vals = {
                "是否推荐": is_rec,
                "推荐译法": rec_trans,
                "权威定义": auth_def,
                "双语示例": bilingual_eg,
                "其它可接受译法": other_trans,
                "不推荐译法": no_rec_trans,
                "备注说明": remark_val,
                "数据来源与链接": source_link,
            }
            for k, val_list in merged_records[domain_val].items():
                current_v = current_vals[k]
                if current_v != "--" and current_v not in val_list:
                    val_list.append(current_v)

    rows_data = []
    clean_zh_display = clean_val(zh_display)
    clean_en_display = clean_val(en_display)

    for domain_k, v_dict in merged_records.items():
        row = {
            "中文术语": clean_zh_display,
            "使用领域": domain_k,
            "是否推荐": (
                "; ".join([x for x in v_dict["是否推荐"] if x != "--"])
                or "--"
            ),
            "推荐译法": (
                "; ".join([x for x in v_dict["推荐译法"] if x != "--"])
                or "--"
            ),
            "权威定义": (
                " \n ".join([x for x in v_dict["权威定义"] if x != "--"])
                or "--"
            ),
            "双语示例": (
                " \n ".join([x for x in v_dict["双语示例"] if x != "--"])
                or "--"
            ),
            "其它可接受译法": (
                "; ".join([x for x in v_dict["其它可接受译法"] if x != "--"])
                or "--"
            ),
            "不推荐译法": (
                "; ".join([x for x in v_dict["不推荐译法"] if x != "--"])
                or "--"
            ),
            "备注说明": (
                " \n ".join([x for x in v_dict["备注说明"] if x != "--"])
                or "--"
            ),
            "数据来源与链接": (
                " | ".join([x for x in v_dict["数据来源与链接"] if x != "--"])
                or "--"
            ),
        }

        if is_verify_mode:
            row = {
                "中文术语": row["中文术语"],
                "输入的英文术语": clean_en_display,
                **{k: v for k, v in row.items() if k != "中文术语"},
            }

        rows_data.append(row)

    df = pd.DataFrame(rows_data)

    html_table = df.to_html(
        index=False, escape=False, classes="custom-rendered-table"
    )
    st.markdown(html_table, unsafe_allow_html=True)
else:
    st.info("💡 暂无检索数据。请在上方输入面板填写术语并点击【🚀 检索】。")
