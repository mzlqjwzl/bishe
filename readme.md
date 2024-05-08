# 图像字幕大师 ImageCaptioning


## 说明


## 数据格式

```
数据集	下载链接
2017 Train images [118K/18GB]	http://images.cocodataset.org/zips/train2017.zip
2017 Val images [5K/1GB]	http://images.cocodataset.org/zips/val2017.zip
2017 Test images [41K/6GB]	http://images.cocodataset.org/zips/test2017.zip
2017 Train/Val annotations [241MB]	http://images.cocodataset.org/annotations/annotations_trainval2017.zip

2014 Train images [83K/13GB]		http://images.cocodataset.org/zips/train2014.zip
2014 Val images [41K/6GB]			http://images.cocodataset.org/zips/val2014.zip
2014 Test images [41K/6GB]			http://images.cocodataset.org/zips/test2014.zip
2014 Train/Val annotations [241MB]  http://images.cocodataset.org/annotations/annotations_trainval2014.zip
2014 Testing Image info [1MB]		http://images.cocodataset.org/annotations/image_info_test2014.zip

————————————————
版权声明：本文为CSDN博主「nanyidev」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/ji_meng/article/details/124959983
```

MS COCO dataset:http://cocodataset.org/

```
{"70570": {"filename": "COCO_val2014_000000323288.jpg", "caption": "\u4e00\u4e2a\u5927\u8001\u4eba\u5750\u4e00\u4e2a\u957f\u6728\u51f3\u5728\u516c\u56ed\u3002", "caption_en": "A large older man sitting on a wooden bench in a park."}, 
"822590": {"filename": "COCO_val2014_000000453104.jpg", "caption": "\u957f\u9888\u957f\u9888\u9e7f\u5728\u5929\u7a7a\u4e2d\u62ac\u8d77\u5934", "caption_en": "a long necked giraffe raises its head in the sky "}} 

{
  "70570": {
    "filename": "COCO_val2014_000000323288.jpg",
    "caption": "一个大老人坐一个长木凳在公园。",
    "caption_en": "A large older man sitting on a wooden bench in a park."
  },
  "822590": {
    "filename": "COCO_val2014_000000453104.jpg",
    "caption": "长颈长颈鹿在天空中抬起头",
    "caption_en": "a long necked giraffe raises its head in the sky "
  }
}
```


简要说明：


模型不用训练，直接加载训练好的

然后，后面的 验证集和测试集 也不要
也不要点

直接到最后一步：实际应用

就可以预测了

预测的图放到test目录下了

你试一下就知道了

这个模型只跑了70轮左右，准确率86.5；

如果作为练习的话，可以让学生们 在这个基础上，继续训练


可西哥  9:34:02
他这个实验的名称，离最后的实现，还差那么一小点

-----------------------------------------
## 扩展思路 

作业一：
模型名字叫“图像字幕”，但是并没有从某段视频里提取图像，然后生成字幕

你可以找一段比较适合的，比如变化比较大的，没有字幕的，短视频，
然后让学生生成字幕，再生成一个带字幕的新视频；

作业二：
另外，这个代码训练数据用的是2014版的，相对比较旧一点, 可以换用2017版本
另外，原始数据是英文版的，然后不知道用了啥工具去翻译成中文的，翻译得那叫一个烂
也可以让学生重新用google或者baidu翻译去重新翻译一下中文，然后再跑模型


-----------------------------------------

## 视频实时字幕


下载视频：
（无水印）唯美治愈素材分享 第10期_哔哩哔哩_bilibili  https://www.bilibili.com/video/BV1BZ4y1i7Hv/

视频下载工具网站：
https://www.bilibilihelper.com/


视频存放：`video/757244476-1-16.mp4`

命令行：
```
python ImageCaptioning.py
```

每隔10秒生成一次字幕，生成的字幕如下：

```

字幕君:['一群鸟在空中飞舞。']，当前帧：1
(360, 640, 3) (244, 244, 3)
字幕君:['一个人在空中跳滑板']，当前帧：50
(360, 640, 3) (244, 244, 3)
字幕君:['一个人坐在长椅上，看着一些树木']，当前帧：650
(360, 640, 3) (244, 244, 3)
字幕君:['一个人在一个大的森林里骑着一只大象。']，当前帧：1250
(360, 640, 3) (244, 244, 3)
字幕君:['一个人走在一条街上，一边有一个停车标志。']，当前帧：1850
(360, 640, 3) (244, 244, 3)
字幕君:['一只鸟站在一些树枝旁边']，当前帧：2450
(360, 640, 3) (244, 244, 3)
字幕君:['一个人在树林里拿着一把伞。']，当前帧：3050
(360, 640, 3) (244, 244, 3)
字幕君:['一只鸟坐在树枝上，看着相机。']，当前帧：3650
```

