# 安装

[Typora 官方中文站 (typoraio.cn)](https://typoraio.cn/)

略.........

# 激活

[Typora激活 (github.com)](https://github.com/743859910/Typora_Unlocker)

1. 将"license-gen.exe"与"node_inject.exe"复制粘贴到Typora安装目录下。
2. 在Typora安装目录中打开cmd窗口或PowerShell窗口。
3. 在窗口命令行输入"node_inject.exe"并回车，运行后显示"done!"。
4. 在窗口命令行输入"license-gen.exe"并回车，复制生成的License。
5. 打开Typora输入生成的License，邮箱任意填写即可。

# 偏好设置

1. “文件”选项：勾选“大纲”——“允许折叠和展开侧边栏的大纲视图”。
2. “Markdown”选项：勾选“Markdown扩展语法”的全部选项。

# 上传图片

## Github

1. [GitHub](https://github.com/)新建一个Repository——如命名为Pictures。
2. 获取Token——点击右侧个人头像--->点击"Setting"--->选择"Developer settings"--->在"Personal access tokens"中选择"Token(classic)"--->在"Generate new token"中"Generate new token(classic)"--->"Note"任意填写、"Expiration"选择"no expiration date"、"Select scopes"重点勾选"repo"选项，完成后点"Generate"即可生成，复制并保存此Token。

## PicGo

1. 下载并安装：[PicGo (github.com)](https://github.com/Molunerfinn/PicGo)
2. 配置Github图床。

![image-20240730142608370](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/image-20240730142608370.png)

## Typora

1. 进入Typora的”偏好设置“——”图像“选项。
3. “插入图片时”选项：选择”上传图片“，并勾选“对本地位置的图片应用上述规则”。
4. “上传服务设定”选项：选择"PicGo(app)"，并选择PicGo路径（找到PicGo.exe的位置）。
5. 点击“验证图片上传选项”，出现如下效果即代表成功。

![image-20240730142532654](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/image-20240730142532654.png)
