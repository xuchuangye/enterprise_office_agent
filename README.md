# Agent的应用场景
1. 深夜紧急报告
2. 知识库问答
3. 自动化报告生成
4. 数据分析与可视化

# 传统程序与Agent
1. 传统程序只能做预先设定好的任务，不会思考，不会变通
2. Agent具备自主规划能力，会思考，会决策，会反思

# Agent的核心能力
自主规划 -> 工具调用 -> 反思迭代
planner.py 拆解复杂任务，规划合理的执行路径与步骤，将大目标转化为可落地的子任务序列
tool_registry.py 统一管理并维护可用的外部能力与工具库，根据任务需求智能匹配并动态调用相应工具
memory_manager.py 持久化存储并高效检索历史对话、关键事实与上下文信息，确保Agent具备长短期记忆能力
reflection.py 复盘并评估过往执行的质量与逻辑，识别潜在错误与改进空间，实现自我修正与持续迭代

# 项目技术栈
LangChain：核心框架/应用集成，只是一个工具箱，提供统一的接口切换各种大模型、工具和组件，只是封装，不改变底层逻辑
通义千问：大语言模型/核心引擎
Streamlit：前端交互/快速原型
FastAPI：后端服务/高性能API
Pandas：数据处理/分析计算
Matplotlib：数据可视化/图表展示
ChromaDB：向量数据库/检索增强
Redis：内存数据库/缓存加速
MySQL：关系型数据库/持久存储

# 系统架构
前端界面层： Streamlit Web Interface 用户交互入口
后端服务层：FastAPI Application HTTP请求与路由分发
Agent核心层：任务规划/工具注册 记忆与反思机制
工具层：数据分析/文档生成 知识库问答/外部API
数据存储层： MySQL/Redis ChromaDB/本地文件

## 全链路处理流程
用户在前端界面输入请求 -> FastAPI后端服务接收并分发 -> Agent核心层分析任务并制定计划 -> 调用适配的工具处理数据 -> 返回结果给用户展示

# LLM基础与通义千问接入
主流模型对比：通义千问，中文语境表现优秀，适合国内应用场景。GPT系列，模型能力强，生态完善。Claude系列，长文本处理能力突出

## LangChain核心组件
Chains ：链，串联多个独立的组件组合成逻辑处理链，将简单的LLM调用转化为复杂的自动化工作流
Prompts：提示词，木版年画管理发送给模型的指令，通过变量填充实现动态的上下文生成
Output Parsers：解析器，将 LLM返回的纯文本转换成JSON、列表等结构化数据，便于后续逻辑处理

# Agent概念与ReAct框架

## 什么是Agent
Agent是一个能够自主感知环境、做出决并执行行动的智能体

在该项目智能办公助手中，Agent扮演着“核心大脑”的角色：理解用户的自然语言意图，拆解复杂的任务逻辑，并调度各类工具去循环迭代返回满意的结果来完成任务

## ReAct框架：思考与行动的闭环
核心思想：打破传统思考与行动的割裂，将逻辑推理与工具调用紧密结合，通过循环逼近目标

1. 思考（Think）：基于输入和历史，决定下一步行动
2. 行动（Act）：调用工具（搜索/计算/API）执行操作
3. 观察（Observe）：获取并记录工具执行后的结果反馈
4. 重复（Repeat）：将观察结果纳入思考，知道任务完成

## ReAct模板
这个ReAct模板就会告诉大模型，如何行动，如何观察，何时结束
```python
    REACT_TEMPLATE = """
    你是一个企业智能办公助手，能够使用各种工具帮助用户完成办公任务。

    你可以使用的工具有：
    {tools}

    工具名称列表：{tool_names}

    请严格按照以下格式回答：

    Question: 用户的问题
    Thought: 思考需要做什么
    Action: 要使用的工具名称（必须是 [{tool_names}] 中的一个）
    Action Input: 工具的输入参数（JSON格式）
    Observation: 工具返回的结果
    ... (这个 Thought/Action/Action Input/Observation 可以重复多次)
    Thought: 我现在知道最终答案了
    Final Answer: 对用户的最终回复

    开始！

    历史对话：
    {chat_history}

    Question: {input}
    Thought: {agent_scratchpad}
```

## 简单任务与复杂任务
简单任务使用ReAct，复杂任务使用规划器

### 为什么使用双路径的执行策略？
简单任务和复杂任务的处理方式应该是不同的，
简单任务基本上只需要一次工具调用，ReAct可以快速完成，不需要复杂规划
复杂任务基本上需要多个步骤，需要提前规划，可能需要多次迭代

### 如何判断是否是复杂任务？
通过关键词列表，只要包含关键词列表中的任意一个关键词，那么就认为是复杂任务，
```python
complex_keywords = ["分析", "报告", "生成", "图表", "可视化", "工作总结", "会议纪要", "审批"]
```
### 自动选择机制
这种设计方式提供智能选择，保留手动控制的灵活性：
1. 用户没有明确指定任务类型，用户传参use_planner=False
    1. 判断规划器已经初始化，并且任务复杂，那么就开启规划器
    2. 否则就使用ReAct Agent
2. 用户明确指定任务类型，也就是用户直接传参use_planner=True，那么就强制使用规划器


# 任务规划器设计

## 规划器的核心价值
将一个复杂、模糊的用户自然语言请求，拆解为一系列逻辑严密、可被Agent执行的、有序的工具调用步骤，是连接用户意图与最终结果的关键桥梁

## 规划器的核心作用
提高效率、提升可靠性、增强可解释性

## 规划器是什么
就像一个项目经理，它的职责是理解任务、拆解任务、分配资源、输出计划

## 为什么需要规划器
没有规划器，Agent需要多次思考、调用工具、观察，主要问题：效率低，走弯路，不可靠
如果有规划器，Agent按照预定的规划执行即可，优势：效率高，步骤清晰，结果可靠

## LLM动态规划模式
核心逻辑： 充分利用大语言模型的上下文理解与推理能力，根据用户输入动态生成步骤
特点：灵活度极高，擅长处理非标、复杂的突发需求，但受LLM性能波动的影响，存在一定的不确定性

## 预设规则的规划模式
针对LLM动态规划模式的缺陷，使用预设规则规划模式进行兜底，具有可靠性和快速响应
核心逻辑：基于预设的业务逻辑与关键词精准匹配，直接映射到固定的执行流程
特点：执行稳定可靠、响应速度快，适合处理高频、标准化的业务场景，但对未覆盖的边缘场景适应性较弱
例如：分析/报告生成/会议纪要/工作总结(周报/月报)/审批(申请)
针对预设规则规划模式的缺陷，还是需要LLM动态规划模式进行结合


### 最终方案
首先尝试LLM动态规划，如果大模型规划成功，就返回分NOE，使用大模型的结果
否则降级到预设规则的规划，进行优雅降级

降级策略能够保证系统的健壮性

## 文件路径的提取
```python
        patterns = [
            # 按照顺序从上往下依次进行文件路径的匹配，返回匹配的第一个结果，找到就立即返回
            # 如果都没有匹配到，那么直接使用默认的文件路径

            # 1）显式标注：文件路径：xxx / path: xxx / 文件：xxx
            r'文件路径\s*[：:]\s*([^\s：:]+)',
            r'path\s*[：:]\s*([^\s：:]+)',
            r'文件\s*[：:]\s*([^\s：:]+)',

            # 2）动词 + 路径（要求中间必须有 / 或 \，防止吞掉前面的中文）
            r'(?:分析|查看|处理|使用|打开|读取)\S*?\s*([\w./\\-]+\.(?:xlsx|xls|csv))',

            # 3）纯文件名（无目录的情况）：只允许字母数字下划线横线 + 点
            r'\b([\w\-]+\.(?:xlsx|xls|csv))\b',

            # 4）完整路径兜底：必须含至少一个 / 或 \，可选盘符
            r'((?:[a-zA-Z]:)?[\w.\-]+(?:[/\\][\w.\-]+)*\.(?:xlsx|xls|csv))',
            r'([^\s]+\.(xlsx|xls|csv))', # 默认的文件路径，最终兜底
        ]
```

# 工具开发与注册机制

## 工具基类设计（BaseTool）
定义统一的抽象接口，确保所有工具具备一致的调用行为与异常处理能力，是构建工具库的基石

### 为什么需要抽象基类
会导致Agent无法统一调用工具，代码混乱，难以维护，新增工具还需要修改Agent代码
抽象基类就是一个契约，规定所有的工具必须遵循标准，统一接口，所有的工具都有相同的调用方式，规范行为，定义工具的处理逻辑，易于扩展，新增工具只需要继承这个基类即可

### 基类的优势
可扩展性，新增工具只需要继承这个基类
一致性，所有工具的行为是一致的，Agent可以统一调用
可维护性，修改基类就可以影响所有的工具
健壮性，统一的异常处理机制


## 注册中心
简单高效的注册机制：无需复杂配置，实例化工具后一行代码完成注册，Agent即可获得新技能
```python
agent.register_tool(CalculatorTool())
```
### 注册机制的作用
1. 统一进行管理，集中所有注册的工具
2. 动态注册，运行时可以添加或者移除工具
3. 方便查找，通过名称快速获取到工具

## 自定义工具---计算器
继承基类并实现核心的_execute()方法，以计算器为例，只需封装数学表达式的计算逻辑即可
1. 定义参数模型以及工具类（工具类需要继承基类）
2. 工具描述：详细说明知识的计算和使用示例
3. 安全执行，使用限制可用的函数，防止安全风险
4. 错误的处理，捕捉语法的错误等常见错误，包括兜底的异常处理机制

# RAG知识问答链路的流程
1. 读取数据
2. 清洗数据：重复数据、异常值、缺失值、格式错误，保证数据的正确性和准确性
    1. 删除重复的行，避免重复计算
    2. 删除全部为空的行，因为是无价值的数据
    3. 对于数值列，用中位数填充缺失值，使用中位数不影响数据，而且中位数不受极端值的影响，保持数据的分布
        跳过空列或者全为NaN的列
    4. 对于文本列，用字符串“未知”填充，保留记录，便于后续去分析

3. 分析数据
    1. 统计基本信息（全局概览）
    2. 描述性统计：
        数值列统计（均值，标准差，最小值，25%分位数，中位数，75%分位数，最大值）
        3. 相关性分析（如果有多列）各个数值之间的相关系数，
        文本列统计（找到文本列的唯一值的数量）

4. 生成可视化图表
    1. 为每个数值列生成直方图
    2. 如果有多个数值列，生成相关性热力图
    注意事项：可视化必须要开启显示中文和显示负号
    ```python
    # 设置Matplotlib的中文字体和负号显示，确保生成的图表能够正确显示中文和负数
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    ```
5. 生成完整的分析报告
    1. 数据分析报告
    2. 数值分析、相关性分析
    3. 分类统计

# 文档生成工具
1. 生成会议纪要（包含参会人员、议程、决议等）
2. 生成工作总结（周报/月报/年报）
3. 生成分析报告（结合数据分析结果）
4. 支持 LLM 智能内容生成

## 文档生成工具输入参数

##  三种文档生成的方式

# API测试工具
企业内部API调用工具
支持OA/CRM系统的流程审批、状态查询等功能
提供模拟模式和真实API模式

## 支持审批的类型
common_types = ["leave", "expense", "contract", "purchase", "travel", "overtime"]
OA系统：["resign", "training"]
不是OA系统：["contract_sign", "customer_visit"]

### 为什么使用双模式？来判断真实API是否可用
使用模拟API和真实的API这种双模拟机制，让开发和使用都非常灵活
在真实的开发阶段，如果OA系统或者CRM系统还没有部署，或者网络连接调用需要认证
例如：发起审批，但只是想进行测试

但如果对接的是真实的OA系统，可能已经发起了请假，审批，就产生实际的业务影响
那么双模式就允许有个入口，使用模拟数据，方便开发和测试
真实的业务场景，就调用真实的API，用于生产环境

因为在开发过程中，会有测试环境、灰度发布环境、生产环境
在API工具类的_execute()执行方法中会判断传递的参数use_mock是否为True，表示强制开启Mock模拟模式，或者真实的API不存在【not self._is_api_available(system)函数用于检查真实的API是否存在】，那么就使用模拟数据。否则就使用真实的API

在实际场景中，如果是开发调试，那么使用模拟数据进行快速测试
如果功能演示，那么使用模拟数据进行测试，确保每次演示的结果都是一样的
如果集成测试，那么使用真实的数据进行测试
如果生产环境，那么使用真实的API，实际执行业务的流程

### 模拟内部的系统
1. API 工具 (模拟企业内部系统)在/config/settings.py
OA_API_BASE_URL: str = "http://localhost:8000/api/oa"
CRM_API_BASE_URL: str = "http://localhost:8000/api/crm"
2. 可以替换成实际的API

# 知识库与RAG实现

## RAG的核心原理

核心思想：在生成回答前，先从企业私有知识库当中去检索与用户问题最相关的上下文信息，再将这些信息作为“提示词”的一部分提供给大模型，让模型基于检索到的事实生成答案，而非凭空捏造

解决痛点：
1. 事实准确性：有效抑制大模型的幻觉问题，确保回答有迹可循，符合企业真实情况
2. 知识时效性：通过更新底层知识库，无需重新训练大模型，即可获取最新的业务动态


## 加载与分割
将文档切分成合适长度的文本块Chunks

因为企业知识库的文档是多样，PDF、Word、Excel、Txt、Markdown等，文档通常很长，可能会超过模型的上下文长度限制（例如 32K、64K），超出则溢出，模型就不能进行处理了
所以需要分块

### 递归字符分割器
```python
    # 分割符的优先级从高到低，也就是从左往右，优先级依次降低
    # \n\n表示段落分割
    # \n表示行分割
    # 中文标点符号和英文标点符号表示句子分割
    # 空格表示单词分割
    # 空字符串表示字符分割
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
```

### 为社么需要重叠？
如果不重叠，语义就不完整，如果重叠，语义更完整，一般是chunk_size的10%~20%

### 加载器
根据文件扩展名自动选择加载方式，PDF、Word、Excel、Txt、Markdown

## 文本向量化
利用Embedding模型将文本转换为模型能够识别的数值向量

### 为什么模型也存在降级策略？
在生产环境中，本地模型的速度快，不依赖网络，开发环境中，在线模型方便测试
如果本地模型损坏，那么就使用在线模型，自动降级保证可用性

## 向量存储
存入Chroma等专用向量数据库，建立索引

## 为什么使用向量数据库Chroma
轻量级，非常适合RAG应用，开源，免费，易于使用，支持持久化，与LangChain无缝集成

## 相似度检索
用户提问之后，将问题进行向量化，检索出语义最相似的Top—N个文本块

## 智能生成
将检索上下文与用户问题拼接，一起传递给大模型生成准确的回答，甚至是来源

# 记忆与反思

## 记忆管理系统：赋予Agent持续学习与个性化能力
为什么记忆是很重要的？
记忆是Agent“类人化”的核心，它能够让Agent理解上下文，避免重复提问，并基于用户历史行为提供千人千面的个性化服务，构建连贯的对话体验

## 短期记忆
目标：维持当前会话上下文，确保流畅性
实现：以Redis为主存储，故障时降级至本地内存，实现毫秒级读写。
会话的上下文，记住当前会话历史，存储的是会话期间，容量是有限的，通常10-20条消息

## 长期记忆
目标：沉淀用户画像、历史偏好及行为数据，赋能个性化推荐
实现：Redis + MySQL组合，兼顾高频查询与数据持久化
用户的偏好，用户画像，类似于人类的长期记忆，存储时间是永久的，容量是无限的

## 核心
短期记忆：存储对话历史，支持多会话隔离，限制最大长度，避免无限增长
长期记忆：存储用户偏好，跨会话持久化，支持快速查询

## 记忆的应用场景
短期记忆：多轮对话，上下文理解
长期记忆：个性化回答，用户的习惯
记忆能够让模型理解上下文，提供更可靠的回答

### 记忆的存储流程
采用三级存储策略，为什么采用三级策略？
1. 存储系统可能会出现各种问题：例如Redis服务的宕机，网络连接失败，MySQL数据库的响应速度慢，如果只依赖单一的存储系统，一旦出现问题，整个系统就会瘫痪

### 三级存储缓存策略
1. 优先访问Redis，最快访问，会话期间有效，如果访问失败，内存进行缓存
2. 但是内存缓存，重启之后数据就会丢失，所以接下来使用MySQL本地数据库，永久存储，跨会话恢复
3. 采用降级缓存：无论Redis是否可用，都维护一个本地列表，短期记忆的缓存，用来应急兜底

### 三级存储缓存策略的优势
1. 高可用性：即使Redis服务宕机，系统仍然可以稳定运行
2. 数据不丢失，MySQL数据库永久存储，保证数据持久化
3. 性能优化：优先使用最快存储Redis
4. 自动恢复，如果Redis宕机和重启之后本地缓存列表消失了，导致都获取不到数据，那么还可以从MySQL中加载历史数据
5. 健壮性

### 三级存储策略的读取流程
1. 如果Redis是有效的，那么就从Redis中获取
2. 如果Redis是无效的，那么就从本地缓存列表中获取
3. 如果本地缓存列表中也没有，那么就从MySQL中获取，并且添加到本地缓存列表

### 在Agent中集成记忆管理器实现记忆的功能
class CustomConversationMemory(ConversationBufferWindowMemory): 扩展记忆类，在保存上下文的时候同步保存到 MemoryManager，并且避免多 key 警告

1. 调用父类ConversationBufferWindowMemory的构造方法，保持LangChain原有的记忆功能
2. 同步用户的输入，将用户的输入保存到记忆管理器当中；同步助手的回答，将助手的回答也保存到记忆管理器当中
3. 集成到Agent当中
```python
        # 创建记忆对象，加载历史对话
        memory = CustomConversationMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input",
            k=settings.SHORT_TERM_MEMORY_MAX_LENGTH,
            memory_manager=self.memory_manager,
            session_id=session_id
        )
```

### 整体记忆的完整流程
1. 创建会话获取到session_id，用户首次输入问题，调用工具生成回答，将用户输入和助手的回答保存到记忆管理器中，采用三级存储策略，存储到Redis或者是MySQL当中
    ```python 
    from .memory_manager import MemoryManager # 引入 MemoryManager记忆管理器

    class CustomConversationMemory(ConversationBufferWindowMemory):
        memory_manager: Optional[MemoryManager] = None
        session_id: str = ""

        def save_context(self, inputs: dict, outputs: dict) -> None:
    ```
2. 用户追问结果，开启第二轮对话，Agent拓展记忆类，开始回忆，在执行器当中从记忆管理器中获取对话历史，理解上下文，拿到对话历史信息的上下文传递给模型思考，再给出准确的回答
    ```python
        # 创建Agent执行器
    def _create_agent_executor(self, session_id: str) -> AgentExecutor:
        # 注册工具
        tools = self.tool_registry.list_tools()
        if not tools:
            raise ValueError("没有注册任何工具，请先注册工具")
        # 提示词模板
        prompt = PromptTemplate.from_template(self.REACT_TEMPLATE)
        # 创建记忆对象，加载历史对话
        memory = CustomConversationMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input",
            k=settings.SHORT_TERM_MEMORY_MAX_LENGTH,
            memory_manager=self.memory_manager,
            session_id=session_id
        )
    ```


## 反思与错误处理
提升系统鲁棒性：智能系统不仅要能成功执行任务，更需要具备”自我修复“的能力，在工具调用失败时，能够分析原因并灵活应对，最终优雅完成目标

### 为什么需要反思？
在实际应用中，任务执行可能会失败，例如：文件路径错误、网络超时、参数不合法、大模型不可用
没有反思模块，失败了一次就放弃了，Agent不会从错误中学习，用户体验很差

如果有反思的模块，Agent会分析错误的原因，尝试修正错误，自动重试，还有自动降级等替代方案
### 失败分析机制
分析单个任务失败原因，返回调整后的任务参数和错误原因
```python
    """
    分析单个任务失败原因，返回调整后的任务参数
    
    Args:
        task: 失败的任务字典
        error: 错误信息
    
    Returns:
        调整后的任务（可能是原参数或修改后的参数）
    """
```
基于任务参数和错误原因进行分析
### 基于大模型的分析和基于规则的分析

```python
    def analyze_failure(self, task: Dict[str, Any], error: str) -> Dict[str, Any]:
        """
        分析单个任务失败原因，返回调整后的任务参数
        
        Args:
            task: 失败的任务字典
            error: 错误信息
        
        Returns:
            调整后的任务（可能是原参数或修改后的参数）
        """
        if self.llm:
            return self._llm_based_analyze(task, error)
        else:
            return self._rule_based_analyze(task, error)

```

#### 基于大模型分析：利用LLM理解并解析错误日志与返回信息

提示词，提供了对工具调用出现错误的重试：
```python
def _llm_based_analyze(self, task: Dict[str, Any], error: str) -> Dict[str, Any]:
        """基于LLM分析"""
        prompt = f"""
        任务执行失败，请分析错误原因并提供修正后的任务参数。
        原始任务：{json.dumps(task, ensure_ascii=False)}
        错误信息：{error}

        请返回一个JSON对象，包含 tool 和修正后的 params。例如：
        {{"tool": "data_analyzer", "params": {{"action": "read_data", "file_path": "data/sales_data.xlsx"}}}}
        确保只返回有效的JSON。
        """
```
也可以根据API调用出现错误写对应的提示词

#### 基于预设规则分析：快速匹配并识别常见错误类型（例如超时、权限不足、参数错误）
如果大模型解析不行，反思失败了，那么就可以降级基于规则的修正
```python
    def _rule_based_analyze(self, task: Dict[str, Any], error: str) -> Dict[str, Any]:
        """基于规则修正"""
        adjusted = task.copy()
        error_lower = error.lower()
        
        if task["tool"] == "data_analyzer":
            if "file" in error_lower and ("not found" in error_lower or "不存在" in error):
                adjusted["params"]["file_path"] = "data/sales_data.xlsx"
            elif "action" in error_lower:
                adjusted["params"]["action"] = "read_data"  # 确保有效action
        
        elif task["tool"] == "api_tool":
            if "system" in error_lower:
                adjusted["params"]["system"] = "OA"  # 默认使用OA
        
        return adjusted
```
后续可以拓展

### 智能重试策略
识别错误类型后，判断其是否属于可重试范畴，如果符合条件，则在最大次数限制内自动重试

```python
    def should_retry(self, error: str, retry_count: int) -> bool:
        """判断是否应该重试"""
        if retry_count >= self.max_retries:
            return False
        # 可以重试的错误类型：超时、连接错误、网络错误、暂时、超时、重试等
        retryable_errors = ["timeout", "connection", "network", "暂时", "超时", "重试"]
        error_lower = error.lower()
        return any(err in error_lower for err in retryable_errors)
```

### 替代计划生成
当错误不可重试，或者重试多次仍未成功时，系统将启动”Plan B“生成逻辑
1. 重新分析当前任务的目标与剩余资源
2. 调用规则模块，生成一个绕过当前障碍的全新执行路径，确保任务最终完成

```python
    def generate_alternative_plan(self, original_tasks: List[Dict], error_context: str) -> List[Dict]:
        """
        当整体任务失败时，生成替代执行计划
        
        Args:
            original_tasks: 原始任务列表
            error_context: 错误上下文
        
        Returns:
            替代任务列表
        """
        if self.llm:
            prompt = f"""
                原始任务计划遇到问题，请生成一个替代方案。
                原始计划：{json.dumps(original_tasks, ensure_ascii=False)}
                错误描述：{error_context}

                请返回一个新的任务步骤列表，JSON数组格式。
                """
            try:
                response = self.llm.invoke(prompt)
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
            except Exception as e:
                logger.error(f"LLM替代计划生成失败: {e}")
        
        # 降级：返回简化计划
        return [
            {"tool": "knowledge_qa", "params": {"question": error_context}}
        ]
```

### 反思的整体流程
1. 如果任务执行失败，先分析错误的类型，是基于大模型的反思还是基于规则的反思
2. 如果是大模型分析，那么就智能生成解决方案
3. 如果是基于规则分析，那么就快速修正固定的规则
4. 不管是大模型反思还是规则反思，都会调整修正参数，获取到修正的结果之后，判断是否重试
    1. 如果是可重试，重新执行，如果成功就返回，如果不成功就生成替代方案，如果LLM生成替代方案失败，那么就回退到知识回答模块进行兜底
    2. 如果判断不可重试，直接返回错误即可

# 前端界面开发---Streamlit
Streamlit只需编写简单的Python代码，即可快速构建并部署美观、交互性强的数据应用和AI原生界面。非常适合Agent应用的快速验证与展示

## 多模式页面设计
利用Streamlit的侧边栏(Sidebar)功能，可轻松实现功能模块的快速切换。
典型的企业级应用场景可以包含：
1. 智能聊天模式：与Agent进行自然语言对话
2. 知识库管理：上传、预览和管理私有文档
3. 工具测试：独立测试各个Function Call工具

## 聊天界面的核心逻辑---状态管理机制
```python
# 会话状态管理的初始化
# session_state状态管理机制，类似于浏览器的本地缓存
# 没有状态管理机制，用户输入会被清空，聊天历史会丢失，上传的文件会消失

# 状态管理的流程：
# 1. 交互，点击按钮，输入文本，Streamlit重新运行脚本，从session_state中读取状态，更新界面，保存新状态到st.session_state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_" + str(uuid.uuid4())[:8]
# 状态管理器的最佳实践：如果messages没有被状态管理，就添加管理，这是存储会话历史    
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None

```
## 新建会话时---存储会话历史
```python
    if st.button("🔄 新建会话"):
        agent.clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        # 如果messages没有被状态管理，那么就添加状态管理，这就是存储会话历史
        st.session_state.messages = []
        st.session_state.uploaded_file_path = None
        st.rerun()
```


## 智能对话---从状态管理中获取对话历史信息
```python
if mode == "💬 智能对话":
    # 显示历史消息（Markdown 格式）
    # 因为有状态管理，所以会从状态管理中找到存储的历史会话中的会话信息，信息存储按照role角色和content内容进行存储
    # 通过chat_message指定角色，通过markdown渲染文本的输出
    for msg in st.session_state.messages:
        # chat_message的列表：'user', 'assistant', 'ai', 'human'
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

```
## 添加用户---更新状态
```python
    if prompt_text.strip():
        # 添加用户消息，其实就是通过状态管理来更新状态
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)
            if st.session_state.uploaded_file_path:
                st.caption(f"📎 {os.path.basename(st.session_state.uploaded_file_path)}")

```
## 切换会话---清除状态
```python
    # 切换模式时清除上传文件状态
    if mode != "💬 智能对话":
        st.session_state.uploaded_file_path = None
```

## 折叠展示
优势：界面简单，不占用空间，可展开，用户交互体验比较好

```python

    # 执行步骤折叠展示
    # 折叠的优势：界面简单，不占用空间，可展开，用户交互体验比较好
    if result.get("intermediate_steps"):
        with st.expander("📋 查看执行步骤", expanded=False):
            for i, step in enumerate(result["intermediate_steps"], 1):
                # 兼容元组格式（ReAct）和字典格式（规划器）
                if isinstance(step, tuple):
                    action, obs = step[0], step[1]
                    tool_name = action.tool
                    tool_input = action.tool_input if hasattr(action, 'tool_input') else {}
                else:
                    tool_name = step.get("tool", "未知工具")
                    tool_input = step.get("tool_input", {})
                    obs = step.get("observation", "
                st.markdown(f"**步骤 {i}：{tool_name}**")
                if tool_input:
                    st.json(tool_input)
                obs_str = str(obs)
                if len(obs_str) > 300:
                    with st.expander("结果详情"):
                        st.code(obs_str, language="text")
                else:
                    st.code(obs_str, language="text")
                st.markdown("---")
```
## 缓存Agent
```python
# 初始化Agent（使用缓存避免重复初始化）
# 缓存Agent实例，避免每次交互都需要重新初始化
# 缓存向量存储，避免重复加载模型，提升性能，减少启动时间
@st.cache_resource
def init_agent() -> OfficeAgent:
    agent = OfficeAgent()
    agent.register_tools([
        DataAnalyzerTool(),
        DocumentGeneratorTool(),
        APITool(),
        KnowledgeQATool()
    ])
    return agent
```

## 知识库管理---防止重复上传文档
```python
    # 防止重复上传
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()
```

## 用户偏好---永久保存

# API接口开发---FastAPI
核心优势：
1. 基于Python类型注解，代码即文档
2. 自动生成交互式API文档（Swagger UI）
3. 性能极高，可与NodeJS和Go相媲美

## 核心接口设计
```python
from fastapi import FastAPI, HTTPException, UploadFile, File
# BaseModel是一个Python的单体模型，用来针对入参的类型以及格式进行校验
# 自动进行校验输入输出的类型和格式，类型的提示、序列化
from pydantic import BaseModel

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="企业智能办公助手API，支持任务规划、工具调用、知识问答等功能"
)

# 定义请求/响应的实体模型
class TaskRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class TaskResponse(BaseModel):
    output: str
    session_id: str
    intermediate_steps: Optional[List] = []

@app.post("/api/chat", response_model=TaskResponse)
async def chat(request: TaskRequest):
    """处理用户对话请求"""


```
## 接口测试
可通过http://localhost:8000/docs访问自动生成的交互式接口文档进行调试

# Docker容器化部署

## 应用构建脚本---Dockerfile

## 多服务编排---Docker Compose
docker-compose.yml

## 一键启动所有服务
docker-compose up -d

# 总结与展望
1. 学习并掌握了分层Agent架构的核心设计i逻辑，为系统搭建奠定基础
2. 深入实践了LLM集成、任务规划、工具开发、RAG检索增强以及记忆管理等关键技术模块
3. 独立完成了前端交互界面与后端API接口的开发与联调，打通了端到端的链路
4. 生产环境要使用Docker容器化部署，掌握了项目从开发到上线的完整流程

## 未来展望
1. 模型优化：尝试集成更大参数量的开源模型或者基于企业私有数据进行微调，持续提升Agent的理解与生成能力
2. 多模态融合：拓展图像理解、语音交互等多模态能力，支持更丰富的输入输出形式，优化用户体验
3. 深度场景集成：与OA、CRM等企业内部系统打通，实现真正的端到端办公流自动化，最大化降本增效
4. 监控体系建设：建立系统日志与性能监控平台，对Agent的运行进行全方位追踪，保障系统稳定性

# 启动后端
uvicorn interface.fastapi_app:app --host 0.0.0.0 --port 8000

# 启动前端
python -m streamlit run interface/streamlit_app.py --server.port 8501