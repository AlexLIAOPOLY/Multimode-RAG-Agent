# LangChain快速入门与Agent开发实战-Part

1

# 一、LangChain.ai工具生态介绍

本期公开课，我将为大家详细讲解元老级Agent开发工具——LangChain。

# 1. GPT-3时代下第一代大模型开发工具

LangChain 可以称之为自 2022年底 大模型技术爆火以来的第一个真正意义上的大模型开发框架。大 模型本质上无法直接解决实际的问题，仅仅是一个能够分析、推理和生成文本的黑盒。直到现在，所有 的开发者们仍然在不断探索如何把大模型的强大能力与实际应用场景结合起来，而当时 LangChain 的出 现，直接让大模型开发变得简单起来，它将大模型开发过程中常用的功能、工具、流程等等全部封装成 一个个的组件，使开发者可以像搭乐高积木一样，快速的组合出适用于不同场景需求的大模型应用。

LangChain 的首个版本于 2022年10月 开源，直到现在仍然再以一个飞快的速度不断进行迭代升

级。从一个开源 Python/TS 框架逐渐发展，形成包括“链”和“代理”等核心组件，现在已走向企业级阶

段，发展成了 LangChain AI ，其拥有目前 Agent 技术领域最大的开源生态，衍生出了多个开源项目框

架，各自都在大模型的技术领域承担着不同的开发任务角色。

从一个形象的角度来说，LangChain的功能定位其实并不是我们现在所谓的Agent开发框架，而是一 个大模型功能增强器，借助LangChain，哪怕在GPT-3模型时代，也能让模型完成对话、拥有记忆、甚至 是完成结构化输出等功能。

![Image](./pdf_images\page1_img1.png)

LangChain官网：https://www.langchain.com/

# 2. 备受争议的工具功能

虽说LangChain的开源，在短时间内收获了大量的开发者用户，这也一度使得LangChain在2023年 成为最受欢迎的大模型开发工具没有之一。但每项技术都会受限于其诞生的时代背景，LangChain也不 例外。在GPT-3时代，大模型以补全模型为主，只能以类似“成语接龙”的方式对文本进行补全，并且实际 运行效果也非常不稳定。此时LangChain借助一些高层封装的API，能够让模型完成对话、调用外部工 具、甚至是结构化输出等功能，这在当时是非常大的进步，也为开发者提供了极大的便利。

但是，伴随着GPT-3.5模型的发布，对话模型正式登上历史的舞台，并逐渐成为主流。而得益于对话 模型更强的指令跟随能力，很多GPT-3需要借助LangChain才能完成的工作，已经成为GPT-3.5原生自带 的一些功能。而等到GPT-4逐渐普及，包括调用外部工具（Function calling）、结构化输出、系统提示 词等功能，都成了模型的基础功能。而对于开发者而言，此时再使用LangChain再对这些功能进行封装 就显得多此一举。

![Image](./pdf_images\page2_img1.png)

![Image](./pdf_images\page2_img2.png)

![Image](./pdf_images\page2_img3.png)

【考古】补全模型与对话模型对比介绍：https://www.bilibili.com/video/BV1Vk4y1V76h

因此在2023年下半年起的很长一段时间里，LangChain饱受争议，很多开发者觉得LangChain代码 冗余、编写复杂，甚至有开发者觉得LangChain太过于复杂，一个模型调用的过程就涉及到数十个类， 一个项目开发动辄就要用到几十个不同的类，而说明文档更是几百个不同的常用类需要记住，其复杂程 度不亚于学习一门全新的编程语言。

![Image](./pdf_images\page3_img1.png)

而这也使得在某个时间段，LangChain的开发者大规模流失。

举个例子，为了更好的兼容不同模型的调用，谷歌ADK采用了LiteLlm作为底层模型调度框架，一 个库即可调用各类模型，而LangChain则为每个主流模型单独封装了一个库，调用不同模型的时候 需要导入不同模型对应的库，例如调用DeepSeek就需要安装 langchain-deepseek ，而调用 Gemini则需要安装 langchain-google-genai 。

# 3. 更加适用于当前Agent开发的LangChain工具生态

在经历了短暂的阵痛后，LangChain果断进行了大刀阔斧的改革。LangChain调整的思路非常简

单：

1. LangChain本身仍然坚守作为“模型能力增强器”的功能定位，并且逐渐稳定更新节奏和频率，虽说 实际使用LangChain进行开发的代码量仍然没变，但模块划分更加清晰、功能更加丰富和稳定，逐 步达到企业级应用水准。目前最新版LangChain的核心功能如下：

模 块 类 别| 示 例 功 能  
---|---  
模 型 接 口 封 装| OpenAl、Claude、Cohere、Qwen 等 模 型 统 一 调 用 方 式  
输 出 结 构 化| 动 从 模 型 中 解 析 J5ON、Schema、 团 数 签 名 、 文 档 等  
Memory 管 理| Buffer, Summary, Entity, Conversation Memory 等  
Tool 接 入| Web #228, SQL 数 据 库 、Python 执 行 器 、API 代 理 等  
Agent 架 构| ReAct, Self-Ask, OpenAl Function Agent 等 调 度 机 制  
RAG 集 成| 多 种 Retriever、Vector Store、 文 档 拆 分 策 略  
Server/AP1 发 布| 快 速 将 链 部 署 为 Web 服 务 或 A2A Agent  
Debug & Callback| Token 使 用 统 计 、LangSmith 可 视 化 追 踪 等



✅ 所以说：LangChain 是 LLM 功能开发的「积木工厂」，不是简单框架，而是模型增强器 + 应用组装工具箱。

2. 与2023年下半年开源LangGraph，LangGraph作为基于LangChain的更高层次封装，能够更加便捷 的搭建图结构的大模型工作流，也就是现在所谓的Multi-Agent系统，而LangGraph也是目前 LangChian家族最核心的Multi-Agent开发框架。同时可以搭配LangGraph-Studio进行实时效果监 测，实际效果如下所示：

![Image](./pdf_images\page4_img1.png)

![Image](./pdf_images\page4_img2.png)

# 项目官网：https://github.com/langchain-ai/langgraph

需要注意的是，LangGraph底层功能仍然是基于LangChain来实现，简单理解LangGraph本 质上其实就是LangChain的流程调度与智能体编排系统

3. 开源大模型工作流（Agent）可视化监控与测试平台LangSmith，借助LangSmith，开发者能够更 加简单便捷监控基于LangChain生态的Agent运行流程、测试Agent功能和不同提示词等，从而使得 LangChain进一步面向企业级应用开发框架；

LangSmith官网：https://www.langchain.com/langsmith

4. 同时，考虑到LangChain本身较为复杂这一情况，开源了LangChain的“可视化实现版”—— LangFlow，这是一款形式对标Dify、可以通过可视化方式、借助拖拉拽来完成LangChain相同功能 的开发工具。

![Image](./pdf_images\page5_img1.png)

![Image](./pdf_images\page5_img2.png)

同时，这也是目前LangChain工具家族中，仅次于LangChain（109k stars）第二受欢迎的开发框 架，在GitHub上已斩获接近72k stars。

# LangFlow官网：https://www.langflow.org/

同时，相比Dify，LangFlow功能更加完善，并且没有任何商业化的计划，可长期稳定使用。

目前LangChain工具生态如图所示：

![Image](./pdf_images\page6_img1.png)

![Image](./pdf_images\page6_img2.png)

此外，作为底层框架本身，LangChain本身仍保持了非常快速的迭代速度，例如就在5月29号， LangChain发布最新版本，新增了目前通用智能体项目非常需要的沙盒环境（SandBox）：

同时，LangChain也是最早官宣支持谷歌A2A技术协议的开发框架，这也使得LangChain本身的先 进性一直处于整个Agent开发框架的前列。

# 4. 当下大模型开发人员必备技能：LangChain

可以说，经过了近3年的发展，目前LangChain工具生态已经非常全面，能适用于各类不同场景的开 发需求，无论是小规模实验、还是大规模商业化部署，无论是使用代码开发、还是偏向使用低代码开发 工具，LangChain工具家族都能满足开发者的需求。

也正因如此，LangChain可以说是历经大模型Agent技术发展巨变但仍“屹立不倒”的开发工具，哪怕 今年以来OpenAI、谷歌等AI巨头纷纷下场发布全新一代Agent开发框架，但LangChian工具本身仍在很 多场景下不可替代。而相比之下，类似AotuGen、CrewAI等工具的适用面，在大模型技术飞速发展的当 下，正在逐渐减少。截止目前，LangChain仍然是目前大模型开发岗位应用最广的框架没有之一：

![Image](./pdf_images\page7_img1.png)

![Image](./pdf_images\page7_img2.png)

![Image](./pdf_images\page7_img3.png)

同时LangChain也是很多著名的开源项目的底层开发框架，如字节前端时间开源的Deep Research 应用deerflow就是采用了LangChain&LangGraph框架：

项目主页：https://github.com/bytedance/deer-flow

此外如谷歌近期开源的Gemini Fullstack LangGraph Quickstart热门项目，也是使用LangGraph作为基 础框架：

![Image](./pdf_images\page8_img1.png)

## D README 413 Apache-20 license “ 吊 Security

# Gemini Fullstack LangGraph Quickstart

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend agent. The agent is designed to perform comprehensive research on a user's query by dynamically generating search terms, querying the web using Google Search, reflecting on the results to identify knowledge gaps, and iteratively refining its search until it can provide a well-supported answer with citations. This application serves as an example of building research-augmented conversational Al using LangGraph and Google's Gemini models.

项目主页：https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
