# Code Acquisition Attempts

Started: 2026-06-10T13:39:04Z

## git ls-remote: official_github
```text
5ba07ac6661db573af695b419a7947ecb704690f	HEAD
exit_status=0
```

## git ls-remote: gitcode_mirror
```text
548a52bbb105518058e27bf34dcf90bf6f73681a	HEAD
exit_status=0
```

## curl HEAD/range probe: github_codeload_tar
```text
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100 10421    0 10421    0     0  17655      0 --:--:-- --:--:-- --:--:-- 17632100 4428k    0 4428k    0     0  3209k      0 --:--:--  0:00:01 --:--:-- 3208k100 12.4M    0 12.4M    0     0  5944k      0 --:--:--  0:00:02 --:--:-- 5947k
http=200 size=13080980 speed=6087090 url=https://codeload.github.com/real-stanford/diffusion_policy/tar.gz/refs/heads/main
exit_status=0
```

## curl HEAD/range probe: github_archive_zip
```text
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 25272    0 25272    0     0  20914      0 --:--:--  0:00:01 --:--:-- 20914100 7902k    0 7902k    0     0  3604k      0 --:--:--  0:00:02 --:--:-- 8006k100 12.7M    0 12.7M    0     0  5018k      0 --:--:--  0:00:02 --:--:-- 9376k
http=200 size=13332383 speed=5138563 url=https://codeload.github.com/real-stanford/diffusion_policy/zip/refs/heads/main
exit_status=0
```

## curl HEAD/range probe: ghfast_archive_zip
```text
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0  0     0    0     0    0     0      0      0 --:--:--  0:00:01 --:--:--     0100  111k    0  111k    0     0  44315      0 --:--:--  0:00:02 --:--:-- 44313100 4095k    0 4095k    0     0  1145k      0 --:--:--  0:00:03 --:--:-- 1144k100 12.4M    0 12.4M    0     0  2756k      0 --:--:--  0:00:04 --:--:-- 2756k100 12.7M    0 12.7M    0     0  2821k      0 --:--:--  0:00:04 --:--:-- 3456k
http=200 size=13332383 speed=2889274 url=https://ghfast.top/https://github.com/real-stanford/diffusion_policy/archive/refs/heads/main.zip
exit_status=0
```

## curl HEAD/range probe: ghproxy_archive_zip
```text
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100 12946    0 12946    0     0  12796      0 --:--:--  0:00:01 --:--:-- 12805100 1476k    0 1476k    0     0   755k      0 --:--:--  0:00:01 --:--:--  755k100 10.8M    0 10.8M    0     0  3817k      0 --:--:--  0:00:02 --:--:-- 3817k100 12.7M    0 12.7M    0     0  4189k      0 --:--:--  0:00:03 --:--:-- 4190k
http=200 size=13332383 speed=4289763 url=https://gh-proxy.com/https://github.com/real-stanford/diffusion_policy/archive/refs/heads/main.zip
exit_status=0
```

## curl HEAD/range probe: gitcode_archive_zip
```text
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100  1024    0  1024    0     0   3943      0 --:--:-- --:--:-- --:--:--  3953
http=206 size=1024 speed=3943 url=https://gitcode.com/gh_mirrors/di/diffusion_policy/repository/archive.zip
exit_status=0
```

## official codeload archive download
```text
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100 2829k    0 2829k    0     0  2272k      0 --:--:--  0:00:01 --:--:-- 2270k100 12.4M    0 12.4M    0     0  6065k      0 --:--:--  0:00:02 --:--:-- 6065k
http=200 size=13080980 speed=6211105 url=https://codeload.github.com/real-stanford/diffusion_policy/tar.gz/refs/heads/main
archive=/root/FinalProject/official_reproduction/diffusion_policy_main.tar.gz
sha256=c63f8d56279b85d622e0ecb4984c4dcf8b04231d59d10aab446747127bb61835
top_level_files=20
```

