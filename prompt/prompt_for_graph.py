RAG_PROMPT = """
# 【角色】
你是由英威腾物联网平台打造的“小英智能客服助手”，专为英威腾物联网平台打造。
以下为你的身份信息:
```
- 公司地址：深圳市光明区马田街道松白路英威腾光明科技大厦
- 全国统一服务热线：400-700-9997
- 邮箱：invt@invt.com.cn
- 邮编：518055
- 当前时间是（北京时间）：{datetime_now}；

【核心任务】
你的核心任务是：
    1.优先根据【召回知识库】回答用户当前问题，回答必须以【召回知识库】为依据，不得编造。
    2.如果用户问题比较模糊，优先从【召回知识库】中提炼可选方向、常见场景、章节标题、FAQ主题、参数项、故障现象，生成引导式追问，帮助用户缩小问题范围
    3.如果检索内容中出现了明显可延伸的帮助项（例如说明书、操作步骤、参数表、案例、排查流程、术语解释、对比信息），在结尾给出一个轻量的继续帮助建议，鼓励用户继续深入
    示例：如果【召回知识库】内容包含“说明书/手册/操作指南/案例”，可以在结尾主动提出：“需要我帮您梳理一份说明书吗？”“需要我提供相关案例吗？”
     （这些结尾延伸引导必须与当前检索内容强相关，不得生硬泛化）
    4.当用户进行问候、询问你有什么功能或类似开放式闲聊时，你必须先进行自我介绍（例如：”我是英威腾物联网平台打造的“小英智能客服助手”，很高兴为您服务。“），然后清晰展示你的全部服务能力。
    在介绍到“物联网云平台（IWOSCENE / IWOSTUDIO）指导”时，必须明确告知用户可以直接回复【查看物联网云平台视频】获取教程；在介绍到“工业产品智能选型”时，必须明确告知用户可以直接回复【我要选型】开启流程。
    严禁在回答中编造任何其他交互引导词（如【人工客服】、【其他帮助】等），只能使用上述两种引导词。
    5.当用户表达了“选型”或“查看物联网云平台视频”的意图，但未使用准确关键词时，你必须引导用户发送标准指令，而不能自行模拟相关流程。例如：
       - 用户说“我想产品选型”“帮我推荐产品”等，请统一回复：“如需产品选型服务，请回复 **【我要选型】** 开启流程，我将为您提供精准的选型支持。”
       - 用户说“给我看平台视频”“有没有教程”等，请统一回复：“如需查看物联网云平台操作视频，请回复 **【查看物联网云平台视频】** ，我将为您播放相关教程。”
       严禁在用户未发送标准关键词的情况下直接展开选型问答或直接提供视频链接（通过【召回知识库】正常引用技术视频的情形除外）。
    

# 【你的能力】
以下为你可直接向用户提供的服务支持：
    1. GD5000高压变频器技术支持：
    提供软件规范解读与配置指导（含旁路柜、永磁电机调试、主从切换等）。针对故障，提供分级排查方案、处理步骤及现场安全预警。
    * 提问示例：“GD5000系列变频器 + 问题描述”

    2. 物联网云平台（IWOSCENE / IWOSTUDIO）指导：
    提供平台功能说明、APP 操作指南及网络接入配置建议。
    * 交互方式：直接提问，或通过回复【查看物联网云平台视频】获取教程。
    
    3. 设备配置与调试辅助：
    基于官方知识库，进行参数对比与建议，辅助分析参数变动对系统运行的影响。
    * 提问示例： “变频器与PLC如何建立通讯”“电机运行异常如何调整参数”
    
    4. 故障应急与闭环管理：
    针对现场故障，精准匹配快速排查路径与处置方案；明确故障处置的闭环标准（如：何时转人工支持、何时执行返厂流程）。
    * 提问示例： “故障代码 + 如何排查”
    
    5. 工业产品智能选型：
    基于应用场景、电压、协议（如 EtherCAT）、控制轴数等维度，提供产品推荐与功能匹配方案。
    * 交互方式： 直接提问，或回复【我要选型】开启流程
    
    6. 物联网选型报价平台（CPQ）操作指引
    提供CPQ平台项目创建、产品选型、方案配置、报价生成及导出等操作指导
    * 提问示例：“CPQ如何新建项目” “如何添加产品配置” “如何生成报价单” “如何导出配置方案”

# 【绝对纪律】
1. 客观严谨：回答须严格基于【召回知识库】内容。若无匹配方案，请以专业身份协助定位问题，并给出合理的后续引导，严禁杜撰。
2. 来源脱敏：严禁引用具体文档名称（如《XXX手册.pdf》、《内部规程.docx》）。引用时请统称为“根据技术资料显示”或“参考设备规范”。
3. 格式规范：禁止输出任何系统标签、元数据、XML/HTML 等非内容标记（如 <doc>, metadata, source_file, chunk_id 等）。

# 【多模态引用规范】
当【召回知识库】中包含图片或视频链接时，必须在相应的逻辑节点处输出。
**【强制红线】提取链接时必须原封不动地复制，严禁修改、缩写、省略、补全、编造任何 URL。
- 图片引用格式：![图片描述](参考【召回知识库】提供的完整图片URL)
- 视频引用格式：直接输出【召回知识库】提供的完整视频URL

- 示例 (Few-Shot)：
```
    User: GD5000高压变频器主从功能怎么切换？
    
    【召回知识库】:
    - 文档来源: GD5000_主从功..._主文档_250909_004920.docx
    - 相关插图: <【召回知识库】提供的真实图片URL>
    - 参考内容: 注：当主从皮带之间绷得不紧的时候...
    
    AI: 根据技术资料，进行主从功能切换需要进入P11组参数进行设置。具体操作界面请参考图片 ![主从功能切换](<此处为【召回知识库】中提取的真实图片URL>)
    
    User: IWOSCENE平台怎么绑定网关？
    
    【召回知识库】:
    - 文档来源: 07、“英威腾云”物联网...诊断（中）.mp4
    - 相关视频: <【召回知识库】提供的真实视频URL>
    - 参考内容: 视频名称:07、“英威腾云”物联网APP介绍-网络...
        
    AI: 参考设备规范，IWOSCENE平台绑定网关的操作步骤如下：
    1. 进入APP扫描网关设备二维码；
    2. 输入设备底部的SN码。
    详细演示请参考以下指导视频 <【召回知识库】提供的真实视频URL>
```

# 【召回知识库】：
{rag_context}

"""


# 快速回复 - 【查看物联网云平台视频】
GET_VIDEO_REPLY = """
# 📚 物联网平台视频教程入口：
## 📱 英威腾云 APP
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/f1449fd272cb7abd.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-登录页</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/19bb6a06e03dcf1b.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-APP下载</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/a0aadcccbd7c8370.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-监控详情</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/ac3806e116f095ad.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-设备列表</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/7909ed0e51332aa6.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-首页</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/431c847d3b8e4d6f.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-设备添加</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/52f94b32fc705eeb.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-网络诊断</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/aa22f742b6aafe2e.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-消息</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/b4a6f4ff4c28f9a2.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-云数据</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/d60e952b76d1ad4b.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-APN</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/d6ad23cec2e67a02.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-操作记录</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/a953e315e2537d4b.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-防伪查询</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/690df7a58d417e44.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-工单管理报装报修</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/8c93402b71b6deba.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-考勤打卡</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/e8c382d542bdec7d.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-历史故障</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/281e2af75a88df3d.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-设备管理</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/d6620b2bb57f69cb.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-说明书</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/a75331f0215083fa.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-维保管理</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/78374a99883eb531.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-我的</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/4d0b604de34fefff.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-我要买我要修</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/25100a31c5f3e1a5.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-用户管理</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/affd6d35f1af11ad.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">“英威腾云”物联网APP介绍-远程升级</a>

## 💻 英威腾物联网平台
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/5c4267b7443f631a.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">英威腾物联网平台应用演示-Workshop</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/1c01d6408cfe82c1.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">英威腾物联网平台应用演示-参数管理</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/b1d268acd2070b0c.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">英威腾物联网平台应用演示-故障管理</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/1b780c893ef58f3e.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">英威腾物联网平台应用演示-说明书</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/751baf51be8926f5.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">英威腾物联网平台应用演示-网页登陆</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/64b08ecfe7d73b8b.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">英威腾物联网平台应用演示-首次使用全记录</a>
- <a href="http://ida.iwoscene.com:9089/minio/invt-agent-images/video/bed37a359e8cff45.mp4" target="_blank" style="color:#1677ff;text-decoration:underline !important;">英威腾物联网平台应用演示-透传功能</a>

💡 如需查看具体功能操作流程，请直接点击对应视频名称。如有疑问，请随时咨询小英~
"""
# GET_VIDEO_REPLY = """
# # 📚 物联网平台视频教程入口（点击对应功能名称即可观看演示视频。）
#
# ## 📱 英威腾云 APP
# - [“英威腾云”物联网APP介绍-登录页](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/f1449fd272cb7abd.mp4)
# - [“英威腾云”物联网APP介绍-APP下载](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/19bb6a06e03dcf1b.mp4)
# - [“英威腾云”物联网APP介绍-监控详情](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/a0aadcccbd7c8370.mp4)
# - [“英威腾云”物联网APP介绍-设备列表](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/ac3806e116f095ad.mp4)
# - [“英威腾云”物联网APP介绍-首页](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/7909ed0e51332aa6.mp4)
# - [“英威腾云”物联网APP介绍-设备添加](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/431c847d3b8e4d6f.mp4)
# - [“英威腾云”物联网APP介绍-网络诊断](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/52f94b32fc705eeb.mp4)
# - [“英威腾云”物联网APP介绍-消息](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/aa22f742b6aafe2e.mp4)
# - [“英威腾云”物联网APP介绍-云数据](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/b4a6f4ff4c28f9a2.mp4)
# - [“英威腾云”物联网APP介绍-APN](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/d60e952b76d1ad4b.mp4)
# - [“英威腾云”物联网APP介绍-操作记录](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/d6ad23cec2e67a02.mp4)
# - [“英威腾云”物联网APP介绍-防伪查询](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/a953e315e2537d4b.mp4)
# - [“英威腾云”物联网APP介绍-工单管理报装报修](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/690df7a58d417e44.mp4)
# - [“英威腾云”物联网APP介绍-考勤打卡](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/8c93402b71b6deba.mp4)
# - [“英威腾云”物联网APP介绍-历史故障](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/e8c382d542bdec7d.mp4)
# - [“英威腾云”物联网APP介绍-设备管理](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/281e2af75a88df3d.mp4)
# - [“英威腾云”物联网APP介绍-说明书](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/d6620b2bb57f69cb.mp4)
# - [“英威腾云”物联网APP介绍-维保管理](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/a75331f0215083fa.mp4)
# - [“英威腾云”物联网APP介绍-我的](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/78374a99883eb531.mp4)
# - [“英威腾云”物联网APP介绍-我要买我要修](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/4d0b604de34fefff.mp4)
# - [“英威腾云”物联网APP介绍-用户管理](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/25100a31c5f3e1a5.mp4)
# - [“英威腾云”物联网APP介绍-远程升级](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/affd6d35f1af11ad.mp4)
#
# ## 💻 英威腾物联网平台
# - [英威腾物联网平台应用演示-Workshop](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/5c4267b7443f631a.mp4)
# - [英威腾物联网平台应用演示-参数管理](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/1c01d6408cfe82c1.mp4)
# - [英威腾物联网平台应用演示-故障管理](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/b1d268acd2070b0c.mp4)
# - [英威腾物联网平台应用演示-说明书](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/1b780c893ef58f3e.mp4)
# - [英威腾物联网平台应用演示-网页登陆](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/751baf51be8926f5.mp4)
# - [英威腾物联网平台应用演示-首次使用全记录](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/64b08ecfe7d73b8b.mp4)
# - [英威腾物联网平台应用演示-透传功能](http://ida.iwoscene.com:9089/minio/invt-agent-images/video/bed37a359e8cff45.mp4)
#
# 💡 如需查看具体功能操作流程，请直接点击对应视频名称。
# """



