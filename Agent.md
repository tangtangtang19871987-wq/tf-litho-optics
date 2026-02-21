项目目标
实现一个Python工具，支持批量读取2D numpy文件，imshow显示，并可通过命令行参数灵活指定多组1D切片（支持横向/纵向、起止范围、索引），命令行参数格式为 --slice "h,10:20:at15" "v,5:30:at20"。

需求细节
支持任意数量的.npy文件，且shape一致。
每个文件imshow显示在第一行。
支持任意数量的1D切片，每个切片通过 --slice 参数指定，格式为：
h,start:end:atY 表示横向切片（第Y行，start到end列）
v,start:end:atX 表示纵向切片（第X列，start到end行）
start、end可省略，表示全范围
每个切片单独一行，自动适配subplot布局。
CLI参数示例：

python3 plot_numpy_images.py file1.npy file2.npy --slice "h,10:20:at15" "v,5:30:at20"
代码需包含：
numpy文件读取
参数解析
切片字符串解析
matplotlib绘图
错误处理
代码需无报错、可直接运行。
期望Agent行为
一次性完成所有代码实现、参数解析、注释、测试脚本生成。
不需要每步都等待用户确认，除非遇到致命错误。
代码风格清晰，注释简明