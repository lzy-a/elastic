#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dsc.client import DscAccessTokenProvider, DscDelegateTokenProvider, DscAuthenticator
from dsc.util.deploy_env import set_env, ENV

if __name__ == '__main__':
    # 特殊说明：python2版本无kconf包，需要用户手动设置当前环境；
    # python3版本，不需要指定环境，指定当前环境可以调用如下方法。
    # set_env(ENV.STAGING)

    # 认证主体
    principal = "liuziyang05/user@kuaishou.com"
    # 密钥
    secret_key = "cba9c2b9f2c74811b909e504a18ece1f"

    # 获取访问token 使用默认有效期
    token = DscAccessTokenProvider.get_token(principal, secret_key)
    # 获取访问token 自定义有效期
    #DscAccessTokenProvider.get_token_with_time(principal, secret_key, 272800000)
    # 若获取token是为了引擎资源（OLAP、kafka）的认证鉴权，则需要指定资源类型为 AuthResourceType.ENGINE
    # 此时即使当前环境为 ENV.STAGING，也会获取线上token
    # 引擎没有staging环境（https://docs.corp.kuaishou.com/k/home/VK0XmLJew8Sk/fcAABudJfvitsHBn8RNN9-iAE）
    #DscAccessTokenProvider.get_token_with_tt(principal, secret_key, 272800000, AuthResourceType.ENGINE)

    # 测试token是否可以认证通过
    DscAuthenticator.authenticate(principal, token, principal, token)
