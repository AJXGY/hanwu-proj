# 寒武纪GPU docker环境使用指南

目前SDK镜像已导入，使用`docker images`命令可以看到镜像名称为

```cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310```

## 创建容器

命令模板

```bash
docker run --rm -e CAMBRUCON_VISIBLE_DEVICES="all" --shm-size '64gb' -e DISPLAY=unix$DISPLAY --net=host --pid=host -v /usr/bin/cnmon:/usr/bin/cnmon -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix -it --privileged --name YOURSELF_CONTAINER_NAME -v $PWD:/home/$USER -v /data:/data  cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310 /bin/bash
```

- 将`YOURSELF_CONTAINER_NAME`修改为您的个性化容器名称
- 如需持久化容器请删除`--rm`参数
- 建议在您自己的`/home/$USER`路径（bash shell中显示的`~`）中执行该命令
- 如有特大文件请上传到宿主机`/data`目录下，此为容器内`/data`目录的挂载点

## 使用持久化的容器

使用`docker ps`查看已启动的容器，使用`docker ps -a`查看所有容器

如果您的容器没有启动，使用`docker start CONTAINER_NAME`来启动

进入容器交互式命令行使用`docker exec -it CONTAINER_NAME bash`

## pytorch环境

容器内已经有了mlu版本的pytorch库，使用方法与cuda版本类似，只需要在使用时将`cuda`关键字替换为`mlu`即可

```bash
>>> import torch
>>> torch.mlu.is_available()
True
```

如需其他库请使用pip安装，**注意**：有此需求时请在创建容器时不要使用`--rm`参数，否则下次仍需从头创建容器重新安装这些库
