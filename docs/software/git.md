# 安装

[Git - Downloads (git-scm.com)](https://git-scm.com/downloads)

略.........（更改安装位置、勾选添加到Path环境变量，之后一直next即可）。

# 测试

打开cmd窗口或PowerShell窗口，输入下面语句看是否正确显示Git的版本。

```bash
git --version
```

若显示Git不是命令，则尝试在环境变量PATH中添加Git的安装目录下的bin子目录（如D:\Git\bin）。

# 配置

打开cmd窗口或PowerShell窗口，输入下面语句设置Git的user相关信息。

```bash
git config --global user.name “gezq0601”
git config --global user.email “ge051799qi@qq.com”
```

完成后可输入下面语句查看Git的配置信息。

```bash
git config --list
```