# 读取原始文件
with open('train_elastic.py', 'r') as f:
    lines = f.readlines()

# 处理每一行
new_lines = []
for line in lines:
    # 如果行开头是'#'且后面是'kafka_warmup()'，则去掉行开头的'#'
    if line.startswith('#') and line.strip() == '# kafka_warmup()':
        new_line = line[1:]
    else:
        new_line = line
    new_lines.append(new_line)

# 将处理后的内容写回原文件
with open('train_elastic.py', 'w') as f:
    f.writelines(new_lines)