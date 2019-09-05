二手房.csv：二手房数据文件
	CATE：城区
	bedrooms：卧室数
	halls：厅数
	AREA：房屋面积
	floor：楼层
	subway：是否地铁房
	school：是否学区房
	price：单位面积房价
	LONG：经度
	LAT：纬度
	NAME：小区名称
	DISTRICT：区域名称

mydata.csv：对二手房数据进行清洗之后的数据文件，用于建模

数据清洗.R：R文件，二手房数据清洗过程，生成报告中所用的建模数据“mydata.csv”

描述和建模.R：R文件，基于“mydata.csv”的描述和建模过程
	