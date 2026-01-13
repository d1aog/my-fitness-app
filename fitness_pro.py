import streamlit as st
import pandas as pd
import sqlite3
import datetime
import os
import hashlib

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
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    st.error(f"环境检测报错: {e}")
    st.stop()

# --- 0. 配置与初始化 ---
st.set_page_config(page_title="AI 健身助手 Pro", page_icon="🏋️", layout="wide")

# === 密码加密工具 ===
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

# === 数据库初始化 (升级版) ===
def init_db():
    conn = sqlite3.connect('fitness_data.db', check_same_thread=False)
    c = conn.cursor()
    # 1. 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS usersTable
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # 2. 训练表 (增加了 username 字段)
    c.execute('''CREATE TABLE IF NOT EXISTS workouts
                 (username TEXT, date TEXT, body_part TEXT, exercise TEXT, weight REAL, reps INTEGER, sets INTEGER)''')
    # 3. 饮食表 (增加了 username 字段)
    c.execute('''CREATE TABLE IF NOT EXISTS diet
                 (username TEXT, date TEXT, food_item TEXT, calories REAL, protein REAL, carbs REAL, fat REAL)''')
    conn.commit()
    return conn

conn = init_db()
c = conn.cursor()

# === 预设动作库 ===
GYM_MENU = {
    "胸": ["平板卧推", "上斜卧推", "哑铃卧推", "器械夹胸", "双杠臂屈伸", "绳索夹胸"],
    "背": ["引体向上", "高位下拉", "杠铃划船", "坐姿划船", "直臂下压", "单臂哑铃划船"],
    "肩": ["坐姿推举", "哑铃侧平举", "俯身飞鸟", "面拉", "杠铃推举", "前平举"],
    "腿": ["深蹲", "硬拉", "腿举(倒蹬)", "哈克深蹲", "坐姿腿屈伸", "俯身腿弯举"],
    "手臂": ["杠铃弯举", "哑铃弯举", "绳索下压", "窄距卧推", "锤式弯举"],
    "核心": ["卷腹", "平板支撑", "悬垂举腿", "俄罗斯转体", "健腹轮"]
}

# --- 1. 登录/注册逻辑 ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

def login_page():
    st.title("🔐 欢迎来到 AI 健身助手")
    
    menu = ["登录", "注册新账号"]
    choice = st.sidebar.selectbox("菜单", menu)

    if choice == "登录":
        st.subheader("请登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type='password')
        if st.button("登录"):
            c.execute('SELECT * FROM usersTable WHERE username = ?', (username,))
            data = c.fetchall()
            if data:
                if check_hashes(password, data[0][1]):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f"欢迎回来, {username}!")
                    st.rerun()
                else:
                    st.warning("密码错误")
            else:
                st.warning("用户名不存在")

    elif choice == "注册新账号":
        st.subheader("创建新账号")
        new_user = st.text_input("设置用户名")
        new_password = st.text_input("设置密码", type='password')
        if st.button("注册"):
            c.execute('SELECT * FROM usersTable WHERE username = ?', (new_user,))
            if c.fetchall():
                st.error("该用户名已被占用")
            else:
                c.execute('INSERT INTO usersTable(username,password) VALUES (?,?)', 
                          (new_user, make_hashes(new_password)))
                conn.commit()
                st.success("注册成功！请前往登录菜单登录。")

# --- 2. 主程序 (只有登录后才显示) ---
def main_app():
    current_user = st.session_state['username']
    
    # 侧边栏显示当前用户
    with st.sidebar:
        st.markdown(f"### 👤 当前用户: **{current_user}**")
        if st.button("注销退出"):
            st.session_state['logged_in'] = False
            st.rerun()
            
        st.markdown("---")
        st.title("⚙️ AI 设置")
        
        mode = st.radio("选择模式", ["☁️ 在线 AI (DeepSeek/Google等)", "💻 本地 AI (Ollama/免费)", "📝 仅记录 (无 AI)"])
        
        llm = None
        embeddings = None
        
        if mode == "☁️ 在线 AI (DeepSeek/Google等)":
            provider = st.selectbox("服务商", ["OpenAI / 中转 (第三方Key专用)", "Google Gemini (官方)", "DeepSeek (官方)"])
            api_key = st.text_input("API Key", type="password")
            
            base_url = "https://api.openai.com/v1"
            model_name = "gpt-4o-mini"
            
            if provider == "OpenAI / 中转 (第三方Key专用)":
                st.info("💡 闲鱼/淘宝 Key 请配置下方参数")
                base_url = st.text_input("接口地址 (Base URL)", value="https://once.novai.su/v1", help="一定要带 /v1 后缀")
                model_name = st.text_input("模型名称", value="gemini-1.5-pro", help="卖家提供的模型名")
                use_local_embed = st.checkbox("✅ 强制使用本地 PDF 引擎 (推荐)", value=True)
            
            if api_key:
                try:
                    if provider == "DeepSeek (官方)":
                        llm = ChatOpenAI(model="deepseek-chat", openai_api_key=api_key, openai_api_base="https://api.deepseek.com")
                        st.caption("✅ DeepSeek 已连接")
                    elif provider == "Google Gemini (官方)":
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                        st.caption("✅ Google Gemini 已连接")
                    else:
                        llm = ChatOpenAI(model=model_name, openai_api_key=api_key, openai_api_base=base_url)
                        if use_local_embed:
                            st.caption("🚀 正在启用本地 PDF 引擎...")
                            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                            st.caption("✅ 本地 PDF 引擎已就绪")
                        else:
                            try:
                                embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=base_url)
                            except:
                                embeddings = None

                except Exception as e:
                    st.error(f"配置出错: {e}")

        elif mode == "💻 本地 AI (Ollama/免费)":
            model_name = st.text_input("本地模型名", "deepseek-r1:1.5b")
            base_url = st.text_input("本地地址", "http://localhost:11434")
            if st.button("连接本地 AI"):
                try:
                    llm = ChatOllama(model=model_name, base_url=base_url)
                    embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
                    st.success("已连接")
                except:
                    st.error("连接失败")

        # === 自动加载内置知识库 ===
        if "vector_db" not in st.session_state:
            st.session_state.vector_db = None
        
        if mode != "📝 仅记录 (无 AI)" and embeddings and st.session_state.vector_db is None:
            st.write("📚 **正在加载知识库...**")
            knowledge_folder = "knowledge"
            if os.path.exists(knowledge_folder):
                pdf_files = [f for f in os.listdir(knowledge_folder) if f.endswith('.pdf')]
                if pdf_files:
                    try:
                        all_docs = []
                        for file in pdf_files:
                            loader = PyPDFLoader(os.path.join(knowledge_folder, file))
                            all_docs.extend(loader.load())
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                        splits = text_splitter.split_documents(all_docs)
                        st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
                        st.success(f"✅ 已加载 {len(pdf_files)} 本书")
                    except Exception as e:
                        st.error(f"加载失败: {e}")

    # --- 主界面 Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🏋️ 训练", "🍽️ 饮食", "📊 看板", "🤖 分析"])

    # === 模块 A: 训练记录 (带用户过滤) ===
    with tab1:
        st.subheader(f"🔥 {current_user} 的快速打卡")
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
            c.execute("INSERT INTO workouts VALUES (?, ?, ?, ?, ?, ?, ?)", 
                      (current_user, str(datetime.date.today()), part_selected, exercise_selected, w_weight, w_reps, w_sets))
            conn.commit()
            st.success(f"已保存: {exercise_selected}")

    # === 模块 B: 饮食记录 (带用户过滤) ===
    with tab2:
        st.subheader("饮食")
        d_input = st.text_input("吃了什么？")
        c1, c2 = st.columns(2)
        if c1.button("直接记录"):
            c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?, ?)", 
                      (current_user, str(datetime.date.today()), d_input, 0, 0, 0, 0)) 
            conn.commit()
            st.success("已记录")

        if c2.button("AI 估算"):
            if not llm: st.error("请先配置 AI")
            else:
                with st.spinner("AI 计算中..."):
                    try:
                        prompt = f"分析食物：'{d_input}'。只返回5个数字用逗号隔开：食物名,热量,蛋白,碳水,脂肪。例：面,300,10,60,5"
                        res = llm.invoke(prompt).content
                        clean = res.replace("`", "").replace("\n", "").strip()
                        parts = clean.split(',')
                        item, cal, prot, carb, fat = parts[0], parts[1], parts[2], parts[3], parts[4]
                        c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?, ?)", 
                                  (current_user, str(datetime.date.today()), item, float(cal), float(prot), float(carb), float(fat)))
                        conn.commit()
                        st.success(f"已记录: {item}")
                    except: st.error("AI 解析失败")

    # === 模块 C: 数据看板 (只看自己的数据) ===
    with tab3:
        st.subheader("数据管理")
        # 核心：SQL语句增加了 WHERE username = ?
        df_w = pd.read_sql_query("SELECT * FROM workouts WHERE username = ?", conn, params=(current_user,))
        if not df_w.empty:
            df_w['vol'] = df_w['weight'] * df_w['sets'] * df_w['reps']
            st.line_chart(df_w.groupby('date')['vol'].sum())
            csv = df_w.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 导出 CSV", csv, "my_data.csv", "text/csv")
        else:
            st.info("暂无数据，快去训练吧！")

    # === 模块 D: AI 分析 (只分析自己的数据) ===
    with tab4:
        st.subheader("教练点评")
        if st.button("生成报告"):
            if not llm: st.warning("请配置 AI")
            else:
                # 只获取当前用户的数据
                user_data = pd.read_sql_query("SELECT * FROM workouts WHERE username = ?", conn, params=(current_user,)).to_string()
                
                if st.session_state.vector_db:
                    retriever = st.session_state.vector_db.as_retriever()
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "基于书籍：\n{context}\n\n分析用户数据：\n{input}\n\n给出建议。"),
                    ])
                    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
                    input_pkg = {"input": user_data}
                else:
                    prompt = ChatPromptTemplate.from_messages([("system", "分析数据："), ("human", "{input}")])
                    chain = prompt | llm
                    input_pkg = {"input": user_data}

                with st.spinner("AI 分析中..."):
                    try:
                        if st.session_state.vector_db: st.markdown(chain.invoke(input_pkg)["answer"])
                        else: st.markdown(chain.invoke(input_pkg).content)
                    except Exception as e: st.error(f"失败: {e}")

# --- 程序入口 ---
if st.session_state['logged_in']:
    main_app()
else:
    login_page()
