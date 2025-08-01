#!/usr/bin/env python3
"""
MinerU文档处理脚本
功能：
1. 将MinerU解析的图片复制到前端public目录
2. 修改MD文件中的图片路径引用
3. 将处理后的MD内容添加到向量库

Author: AI Assistant
Date: 2025-01-31
"""

import os
import shutil
from pathlib import Path
import re
from typing import List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入LangChain相关包
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 配置参数
MINERU_DIR = "MinerU File"
FRONTEND_PUBLIC_DIR = "agent-chat-ui/public"
VECTORSTORE_PATH = "mcp_course_materials_db"
MD_FILE = "full.md"
IMAGES_DIR = "images"

# 新的图片路径前缀（前端访问路径）
IMAGE_PREFIX = "/langchain-course-images"

def setup_directories():
    """创建必要的目录"""
    frontend_images_dir = Path(FRONTEND_PUBLIC_DIR) / "langchain-course-images"
    frontend_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 创建前端图片目录: {frontend_images_dir}")
    return frontend_images_dir

def copy_images_to_frontend(source_dir: Path, target_dir: Path) -> List[str]:
    """复制图片到前端目录"""
    copied_images = []
    source_images_dir = source_dir / IMAGES_DIR
    
    if not source_images_dir.exists():
        print(f"⚠️  图片目录不存在: {source_images_dir}")
        return copied_images
    
    for image_file in source_images_dir.glob("*.jpg"):
        target_file = target_dir / image_file.name
        shutil.copy2(image_file, target_file)
        copied_images.append(image_file.name)
        print(f"📷 复制图片: {image_file.name}")
    
    print(f"✅ 总共复制了 {len(copied_images)} 张图片")
    return copied_images

def update_image_paths_in_md(md_content: str) -> str:
    """更新MD文件中的图片路径"""
    # 匹配图片引用格式: ![](images/filename.jpg)
    pattern = r'!\[\]\(images/([^)]+)\)'
    
    def replace_image_path(match):
        filename = match.group(1)
        new_path = f"{IMAGE_PREFIX}/{filename}"
        return f"![]({new_path})"
    
    updated_content = re.sub(pattern, replace_image_path, md_content)
    
    # 统计替换的图片数量
    original_count = len(re.findall(pattern, md_content))
    print(f"🔄 更新了 {original_count} 个图片路径引用")
    
    return updated_content

def split_markdown_content(content: str) -> List[Document]:
    """分割markdown内容为文档块"""
    # 配置文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n# ",      # 一级标题
            "\n## ",     # 二级标题  
            "\n### ",    # 三级标题
            "\n\n",      # 段落
            "\n",        # 行
            " ",         # 词
            ""           # 字符
        ]
    )
    
    # 分割文本
    texts = text_splitter.split_text(content)
    
    # 创建Document对象
    documents = []
    for i, text in enumerate(texts):
        doc = Document(
            page_content=text,
            metadata={
                "source": "LangChain课程文档",
                "chunk_id": i,
                "document_type": "course_material",
                "processed_by": "MinerU"
            }
        )
        documents.append(doc)
    
    print(f"📄 文档分割为 {len(documents)} 个块")
    return documents

def add_to_vectorstore(documents: List[Document]):
    """将文档添加到向量库"""
    try:
        # 初始化embeddings
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
            base_url="https://ai.devtool.tech/proxy/v1",
            model="text-embedding-3-small",
        )
        
        # 检查是否存在现有向量库
        vectorstore_path = Path(VECTORSTORE_PATH)
        
        if vectorstore_path.exists():
            print("📚 加载现有向量库...")
            # 加载现有向量库
            vectorstore = FAISS.load_local(
                folder_path=VECTORSTORE_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            
            # 添加新文档
            vectorstore.add_documents(documents)
            print("✅ 新文档已添加到现有向量库")
            
        else:
            print("🆕 创建新的向量库...")
            # 创建新的向量库
            vectorstore = FAISS.from_documents(documents, embeddings)
            print("✅ 新向量库创建成功")
        
        # 保存向量库
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"💾 向量库已保存到: {VECTORSTORE_PATH}")
        
        # 显示向量库信息
        print(f"📊 向量库总文档数: {vectorstore.index.ntotal}")
        
    except Exception as e:
        print(f"❌ 向量库操作失败: {str(e)}")
        raise

def validate_environment():
    """验证环境配置"""
    required_keys = ["OPENAI_API_KEY"]  # Please set your API key in .env file
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ 缺少必需的环境变量: {', '.join(missing_keys)}")
        print("请在 .env 文件中配置这些变量")
        return False
    
    print("✅ 环境变量验证通过")
    return True

def main():
    """主函数"""
    print("🚀 开始处理MinerU文档...")
    
    # 验证环境
    if not validate_environment():
        return
    
    # 检查源文件
    source_dir = Path(MINERU_DIR)
    md_file_path = source_dir / MD_FILE
    
    if not md_file_path.exists():
        print(f"❌ 找不到MD文件: {md_file_path}")
        return
    
    try:
        # 1. 设置目录
        frontend_images_dir = setup_directories()
        
        # 2. 复制图片到前端
        copied_images = copy_images_to_frontend(source_dir, frontend_images_dir)
        
        # 3. 读取和处理MD文件
        print(f"📖 读取MD文件: {md_file_path}")
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 4. 更新图片路径
        updated_content = update_image_paths_in_md(md_content)
        
        # 5. 保存更新后的MD文件（可选）
        updated_md_path = source_dir / "full_updated.md"
        with open(updated_md_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"💾 保存更新后的MD文件: {updated_md_path}")
        
        # 6. 分割内容
        documents = split_markdown_content(updated_content)
        
        # 7. 添加到向量库
        add_to_vectorstore(documents)
        
        print("🎉 处理完成！")
        print(f"📝 总结:")
        print(f"   - 复制图片: {len(copied_images)} 张")
        print(f"   - 文档分块: {len(documents)} 个")
        print(f"   - 向量库路径: {VECTORSTORE_PATH}")
        print(f"   - 前端图片路径: {frontend_images_dir}")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()