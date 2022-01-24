# Git命令行使用方法（上传新工程）

之前一直用的都是github官方推荐的GUI软件，但其思路一直是，建立一个新的本地文件夹，然后往里面添加代码文件。查阅了很多方法想直接在本地某个已有文件夹直接新建git本地仓库但都无果（会的旁友教教我），所以最终还是用的命令行实现的，发现也并不难。

## 具体操作

1、查看是否安装git：``git --version``查看。如果没有的话，可以到官方提供的下载，或者通过xcode和homebrew安装。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/git命令行使用方法/image-20220123142115474.png" alt="image-20220123142115474" style="zoom:50%;" />

2、我用的macos系统，没有配置SSH秘钥，直接加了代理，利用``git config``配置了github的用户邮箱。

```bash
git config --global user.name "Your Name"  //配置用户名称

git config --global user.email "Your Email" //配置用户邮箱

git config --list   //查看配置信息
```

3、在网址端新建一个仓库（这步略了）

4、找到本地要相连接的工程，cd到相应文件夹，然后执行：

```shell
git init
```

这时候通过``ls -al``可以查看，文件下是有一个.git文件夹的，这里面的东西尽可能不要碰。

5、添加想要的文件：

```shell
git add . //添加左右新增或者改动过的文件
```

6、标注变更的信息：

```shell
git commit -m "msg"
```

这一步，如果是新文件的话，**一定要有！**，否则会上传不成功。

7、远程仓库和本地仓库连接：

```shell
git remote add origin [url]
```

这里的url就是仓库的网址，复制过来就行。如果链接错了仓库，可以执行删除所连接仓库命令：

```shell
git remote rm origin
```

8、修改分支名称。由于git修改过一次版本后，远程端的主分支不叫master了，而是改叫main，但是本地的主分支还是叫master，所以拉取上传时候会出错（这里也困扰了我一小会），需要执行以下命令更改：

```shell
git branch -m master main
```

9、拉取和上传。一般pull操作就是把远端仓库的拉过来本地仓库中进行整合，push就是再将整合好的上传上去，所以这两个命令一般都是先后要执行的：

```shell
git pull origin main

git push origin main
```

这时候再去看网站端的仓库，应该就都已经上传上去了。

由于我并没有涉及到太多的分支操作，所以也没有相应学习branch的操作指令，以上这些对于新建一个仓库上传来说足够用了，也欢迎小伙伴们补充。