# 图4-2后典型站点讨论的证据记录

段落采用GOB、CAB和DRA三站，保持简短；不改图表和指标。气候背景与测试样本的辐射特征分别核对，不将多年气候直接视为本次天气观测。

## 测试数据

由 scripts/audit_station_weather_20260912.py 对图4-2的同一452条配对测试记录计算。详见 station_weather_context.json。

晴空指数 = 观测GHI / 同小时Haurwitz晴空GHI。数值可能大于1；这里仅用于描述相对辐射水平，不将其作为云量实测或晴天分类标签。起报前指数按对应目标记录加权，便于与SP误差对照。

- GOB：目标时段平均指数1.017，历史平均0.938；目标10%/50%/90%分位数0.931/1.038/1.085。SP nMBE −8.60%，动态融合+1.77%。支持“目标时段整体更接近晴空辐射水平；SP延续较低历史状态而低估”。未认定本次具体案例发生雾消散。
- CAB：目标指数10%/50%/90%分位数0.247/0.622/0.960，表明辐射状态范围较宽；图4-2在4–6 h的RMSE相对降幅增大。云况变化的机制解释用“可能”，未称因果验证。
- DRA：目标指数中位数1.040，SP nMBE −6.54%，动态融合+8.55%。支持“多数目标接近晴空辐射水平，融合存在正偏差”。不概括为全部测试时段晴空。

## 气候背景来源

1. Eckardt, F. D. et al. The nature of moisture at Gobabeb, in the central Namib Desert. *Journal of Arid Environments* 93, 7–19 (2013). https://doi.org/10.1016/j.jaridenv.2012.01.011
   - 已核对全文首页及摘要。研究介绍Gobabeb干旱背景及平流雾、沿海层云等过程。2012年在线发表，卷期年为2013，正文采用2013。
   - 全文：https://www.fogquest.org/wp-content/uploads/2023/06/i-Eckhardt-et-al-2013-The_nature_of_moisture_at_Gobabeb_in_the.pdf
2. Bosveld, F. C. et al. Fifty Years of Atmospheric Boundary-Layer Research at Cabauw Serving Weather, Air Quality and Climate. *Boundary-Layer Meteorology* 177, 583–612 (2020). https://doi.org/10.1007/s10546-020-00541-w
   - 已核对出版社全文：Cabauw为marine west coast climate，Köppen Cfb；论文涉及云、辐射与边界层观测。
3. NOAA Global Monitoring Laboratory. SURFRAD Network: Desert Rock, NV. https://gml.noaa.gov/grad/surfrad/desrock.html
   - 核对内华达站点位置和1007 m海拔。干旱气候分类沿用正文表1；当前辐射状态由上述本次测试记录判断。
