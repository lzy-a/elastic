# 使用官方 Golang 1.12 镜像作为基础镜像
FROM golang:1.12

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到工作目录
COPY . .

# 构建 Go 程序
RUN go build -o main

# 暴露程序需要的端口（如果有需要）
EXPOSE 8080

# 运行 Go 程序
CMD ["./main"]
