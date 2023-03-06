#### Code

目前kernel实现的代码在  `./Attention/layers/kernels` 目录下面，FMHA的实现是 `sparse_attention_new.cu`文件。

`./Attention/python/benchmark.py` 是测试的 python 代码

#### End to end

 的效果如下，我们每一次inference时间是100ms左右。

![image-20230306221936506](https://raw.githubusercontent.com/WongHuLin/picture/main/202303062219397.png)



Flashattn的效果如下，113ms左右：

![image-20230306222152082](https://raw.githubusercontent.com/WongHuLin/picture/main/202303062221686.png)



#### Kernel

FHMA 性能对比，我们大概是4.6ms

![image-20230306222401620](https://raw.githubusercontent.com/WongHuLin/picture/main/202303062224473.png)

Flashattn 大概是4.2ms

![image-20230306222252253](https://raw.githubusercontent.com/WongHuLin/picture/main/202303062222065.png)