#!/bin/bash
# install_sat_solvers.sh - 增强版安装脚本

echo "正在配置 SAT 求解器环境..."

# 检查操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "检测到 Linux 系统..."
    
    # 1. 安装必要的编译工具和依赖
    echo "正在安装编译工具..."
    sudo apt-get update
    sudo apt-get install -y build-essential git cmake zlib1g-dev

    # 2. 安装/编译 Kissat (推荐求解器)
    if ! command -v kissat &> /dev/null; then
        echo "未检测到 kissat，正在尝试从源码编译..."
        
        # 创建临时目录
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR" || exit
        
        # 克隆并编译
        echo "下载 Kissat 源码..."
        git clone https://github.com/arminbiere/kissat.git
        cd kissat || exit
        
        echo "编译 Kissat..."
        ./configure
        make
        
        # 安装到系统路径
        echo "安装 Kissat 到 /usr/local/bin..."
        if [ -f "build/kissat" ]; then
            sudo cp build/kissat /usr/local/bin/
            echo "Kissat 安装成功！"
        else
            echo "错误：Kissat 编译失败。"
        fi
        
        # 清理
        rm -rf "$TEMP_DIR"
    else
        echo "Kissat 已安装。"
    fi
    
    # 3. 安装 Minisat (备用求解器)
    if ! command -v minisat &> /dev/null; then
        echo "正在安装 Minisat..."
        sudo apt-get install -y minisat
    else
        echo "Minisat 已安装。"
    fi

    # 4. 尝试安装 Glucose (可选)
    # Glucose 在新版 Ubuntu 可能缺失，这里仅尝试不强制
    if ! command -v glucose &> /dev/null; then
        echo "尝试安装 Glucose (可能不可用)..."
        sudo apt-get install -y glucose-syrup 2>/dev/null || echo "跳过 Glucose (源中未找到)"
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS 部分保持不变 (Homebrew 通常比较可靠)
    echo "检测到 macOS 系统..."
    if ! command -v brew &> /dev/null; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install kissat minisat
else
    echo "不支持的操作系统: $OSTYPE"
    exit 1
fi

echo "========================================"
echo "SAT 求解器安装流程结束。"
echo "当前可用求解器："
command -v kissat &> /dev/null && echo "✓ kissat (推荐)" || echo "✗ kissat"
command -v minisat &> /dev/null && echo "✓ minisat" || echo "✗ minisat"