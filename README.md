# GitHub的使用

## 创建新项目

1. 进入需要创建项目的文件夹，打开`git bash`
2. `$ git init` 创建项目（本地）
3. `$ git add .` 将文件列表加入上传队列（本地）
4. `$ git commit -m 'name'` 本次提交的名称以及如何对项目进行更改（如果是已有项目的话）
    - 若报以下的错误：
        ```git
            *** Please tell me who you are.

            Run

            git config --global user.email "you@example.com"
            git config --global user.name "Your Name"

            to set your account's default identity.
            Omit --global to set the identity only in this repository.

            fatal: unable to auto-detect email address (got 'taster@DESKTOP-NC3GBP3.(none)')
        ```
        则说明没有创建用户
        - `$ git config --global user.email "you@example.com"` 输入邮箱
        - `$ git config --global user.name "Your Name"` 输入名字
5. `$ ssh-keygen -t rsa -C "you@example.com"` 创建ssh
    之后的三个选项直接回车，可以在显示的路径中看到ssh key
6. 登陆[github](https://github.com/)，创建一个项目
7. `$ git remote add origin https://github.com/you/example.git` 网址中输入新创建的项目地址
    - 若需要更改上传项目，则可以输入
        ```git
            $ git remote rm origin
        ```
        来删除origin
8. `$ git push origin master` 上传项目

## 上传到已有项目

```git
    $ git add .
    $ git commit -m 'name'
    $ git push origin master
```

## 异地更改+同步

当需要在异地（在机器B上）开发该项目的时候，可以使用

```git
    $ git clone https://github.com/you/example.git
```

用来将github上面的项目克隆到机器B

当在机器B上做了一些更改之后，又想回到机器A开发，可以在机器A上对该项目做出的更改进行同步

```git
    $ git pull origin master
```