import streamlit as st
import pandas as pd
import sqlite3
import datetime
import os
import tempfile

# --- 兼容性导入 ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError as e:
    st.error(f"环境检测报错: {e}")
    st.stop()

# --- 0. 配置与初始化 ---
st.set_page_config(page_title="AI 健身助手 Pro", page_icon="🏋️", layout="wide")

# === 预设动作库 ===
GYM_MENU = {
    "胸": ["平板卧推", "上斜卧推", "哑铃卧推", "器械夹胸", "双杠臂屈伸", "绳索夹胸"],
    "背": ["引体向上", "高位下拉", "杠铃划船", "坐姿划船", "直臂下压", "单臂哑铃划船"],
    "肩": ["坐姿推举", "哑铃侧平举", "俯身飞鸟", "面拉", "杠铃推举", "前平举"],
    "腿": ["深蹲", "硬拉", "腿举(倒蹬)", "哈克深蹲", "坐姿腿屈伸", "俯身腿弯举"],
    "手臂": ["杠铃弯举", "哑铃弯举", "绳索下压", "窄距卧推", "锤式弯举"],
    "核心": ["卷腹", "平板支撑", "悬垂举腿", "俄罗斯转体", "健腹轮"]
}

def init_db():
    conn = sqlite3.connect('fitness_data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS workouts
                 (date TEXT, body_part TEXT, exercise TEXT, weight REAL, reps INTEGER, sets INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS diet
                 (date TEXT, food_item TEXT, calories REAL, protein REAL, carbs REAL, fat REAL)''')
    conn.commit()
    return conn

conn = init_db()

# --- 1. 侧边栏：核心设置 ---
with st.sidebar:
    st.title("⚙️ 设置")
    
    # 修改模式名称
    mode = st.radio("选择模式", ["☁️ 在线 AI (DeepSeek/Google等)", "💻 本地 AI (Ollama/免费)", "📝 仅记录 (无 AI)"])
    
    llm = None
    embeddings = None
    
    if mode == "☁️ 在线 AI (DeepSeek/Google等)":
        # === 新增 DeepSeek 选项 ===
        provider = st.selectbox("服务商", ["DeepSeek (深度求索)", "Google Gemini", "OpenAI / 中转"])
        api_key = st.text_input("API Key", type="password")
        
        if api_key:
            try:
                # 1. DeepSeek 配置
                if provider == "DeepSeek (深度求索)":
                    base_url = "https://api.deepseek.com"
                    # DeepSeek-V3 (chat) 是目前的主力模型
                    llm = ChatOpenAI(model="deepseek-chat", openai_api_key=api_key, openai_api_base=base_url)
                    embeddings = None # DeepSeek 暂不支持兼容的 Embeddings 接口
                    st.caption("✅ 已连接 DeepSeek-V3 (注: 暂不支持 PDF 知识库)")

                # 2. Google Gemini 配置
                elif provider == "Google Gemini":
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                    st.caption("✅ 已连接 Google Gemini")

                # 3. OpenAI / 中转 配置
                else:
                    base_url = st.text_input("接口地址", value="https://api.openai.com/v1")
                    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, openai_api_base=base_url)
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=base_url)

            except Exception as e:
                st.error(f"配置出错: {e}")

    elif mode == "💻 本地 AI (Ollama/免费)":
        st.info("请确保电脑已安装 Ollama 并运行了模型")
        model_name = st.text_input("本地模型名", "deepseek-r1:1.5b")
        base_url = st.text_input("本地地址", "http://localhost:11434")
        if st.button("连接本地 AI"):
            try:
                llm = ChatOllama(model=model_name, base_url=base_url)
                embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
                st.success("已连接本地大脑！")
            except:
                st.error("连接失败，请检查 Ollama 是否运行")

    # === 自动加载内置知识库 ===
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    
    # 只有当 embeddings 存在时(非DeepSeek)，才加载知识库
    if mode != "📝 仅记录 (无 AI)" and embeddings and st.session_state.vector_db is None:
        st.markdown("---")
        st.write("📚 **正在加载内置知识库...**")
        knowledge_folder = "knowledge"
        
        if not os.path.exists(knowledge_folder):
             st.warning(f"未找到 {knowledge_folder} 文件夹")
        else:
            pdf_files = [f for f in os.listdir(knowledge_folder) if f.endswith('.pdf')]
            if pdf_files:
                try:
                    all_docs = []
                    for file in pdf_files:
                        file_path = os.path.join(knowledge_folder, file)
                        loader = PyPDFLoader(file_path)
                        all_docs.extend(loader.load())
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                    splits = text_splitter.split_documents(all_docs)
                    st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
                    st.success(f"✅ 已加载 {len(pdf_files)} 本专业书籍！")
                except Exception as e:
                    st.error(f"加载失败: {e}")

# --- 2. 主界面 ---
tab1, tab2, tab3, tab4 = st.tabs(["🏋️ 训练", "🍽️ 饮食", "📊 看板", "🤖 分析"])

# === 模块 A: 训练记录 ===
with tab1:
    st.subheader("🔥 快速打卡")
    part_selected = st.pills("Part", list(GYM_MENU.keys()), default="胸", selection_mode="single")
    exercise_list = GYM_MENU.get(part_selected, ["自定义"])
    exercise_selected = st.pills("Exercise", exercise_list, default=exercise_list[0], selection_mode="single")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: w_weight = st.number_input("重量 (kg)", value=0.0, step=2.5)
    with c2: w_reps = st.number_input("每组次数", value=8, step=1)
    w_sets = st.pills("Sets", [1, 2, 3, 4, 5], default=1, selection_mode="single")
    
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.button("✅ 确认保存", use_container_width=True, type="primary"):
        c = conn.cursor()
        c.execute("INSERT INTO workouts VALUES (?, ?, ?, ?, ?, ?)", 
                  (str(datetime.date.today()), part_selected, exercise_selected, w_weight, w_reps, w_sets))
        conn.commit()
        st.success(f"已保存: {exercise_selected}")

# === 模块 B: 饮食记录 (适配 DeepSeek) ===
with tab2:
    st.subheader("饮食")
    d_input = st.text_input("吃了什么？", placeholder="例如：牛肉面一碗")
    c1, c2 = st.columns(2)
    if c1.button("直接记录"):
        c = conn.cursor()
        c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?)", (str(datetime.date.today()), d_input, 0, 0, 0, 0)) 
        conn.commit()
        st.success(f"已记录: {d_input}")

    if c2.button("AI 估算"):
        if not llm:
            st.error("请先在左侧配置 API Key")
        else:
            with st.spinner("DeepSeek 思考中..."):
                try:
                    # 提示词微调，适应 DeepSeek
                    prompt = f"分析食物：'{d_input}'。请只返回5个数字，用逗号隔开，顺序是：热量(kcal),蛋白(g),碳水(g),脂肪(g)。如果没有食物名，第一项填食物名。例如：'牛肉面,600,25,80,20'。不要任何其他废话。"
                    res = llm.invoke(prompt).content
                    # 简单清洗数据，防止 DeepSeek 话痨
                    clean_res = res.replace("`", "").replace("\n", "").strip()
                    parts = clean_res.split(',')
                    if len(parts) >= 5:
                        item, cal, prot, carb, fat = parts[0], parts[1], parts[2], parts[3], parts[4]
                        c = conn.cursor()
                        c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?)", 
                                  (str(datetime.date.today()), item, float(cal), float(prot), float(carb), float(fat)))
                        conn.commit()
                        st.success(f"已记录: {item} ({cal} kcal)")
                    else:
                         st.error(f"格式解析失败，AI 返回: {clean_res}")
                except Exception as e:
                    st.error(f"AI 没看懂: {e}")

# === 模块 C: 数据看板 ===
with tab3:
    st.subheader("数据管理")
    df_w = pd.read_sql_query("SELECT * FROM workouts", conn)
    if not df_w.empty:
        df_w['vol'] = df_w['weight'] * df_w['sets'] * df_w['reps']
        st.line_chart(df_w.groupby('date')['vol'].sum())
        csv = df_w.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出 CSV", csv, "workout_data.csv", "text/csv")

# === 模块 D: AI 分析 (DeepSeek 适配) ===
with tab4:
    st.subheader("教练点评")
    if st.button("生成报告"):
        if not llm:
            st.warning("请配置 API Key")
        else:
            w_data = pd.read_sql_query(f"SELECT * FROM workouts", conn).to_string()
            
            # 如果是 DeepSeek，跳过知识库检索
            if provider == "DeepSeek (深度求索)":
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一名严厉的专业健身教练。请根据用户的训练数据，给出犀利的点评和改进建议。"),
                    ("human", "{input}")
                ])
                chain = prompt | llm
                input_pkg = {"input": w_data}
                st.caption("🚀 使用 DeepSeek 模型分析 (纯通用知识，不包含 PDF 书籍内容)")
            
            # 其他模型继续使用知识库
            elif st.session_state.vector_db:
                retriever = st.session_state.vector_db.as_retriever()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "参考书本：\n{context}\n\n分析用户数据：\n{input}\n\n给出建议。"),
                ])
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
                input_pkg = {"input": w_data}
            else:
                prompt = ChatPromptTemplate.from_messages([("system", "分析数据："), ("human", "{input}")])
                chain = prompt | llm
                input_pkg = {"input": w_data}

            with st.spinner("AI 正在分析..."):
                try:
                    if provider != "DeepSeek (深度求索)" and st.session_state.vector_db:
                        st.markdown(chain.invoke(input_pkg)["answer"])
                    else:
                        st.markdown(chain.invoke(input_pkg).content)
                except Exception as e:
                    st.error(f"分析失败: {e}")
