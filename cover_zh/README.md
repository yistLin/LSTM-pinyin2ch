### cover_zh

#### 概要
1. 為簡易的文字檔案轉換（簡體轉繁體 以及 繁體轉簡體)

#### 出處
1. http://blog.roga.tw/2011/04/python-%E6%96%87%E5%AD%97%E6%AA%94%E7%B0%A1%E7%B9%81%E8%BD%89%E6%8F%9B%E7%A8%8B%E5%BC%8F/
2. https://code.google.com/p/convert2utf8/

#### 修改
1. 增加 簡體轉繁體 或 繁體轉簡體
2. 原本為讀取檔案改變為讀取資料夾（包含子資料夾）所有檔案的轉換


#### 用法
1. cover_zh.py {資料夾} {轉換語系(tw,cn)} {文件原本編碼}
2. 執行後 將會直接覆蓋 不會做備份。

```
> 轉簡體
python cover_zh.py file cn utf8

> 轉繁體
python cover_zh.py file tw utf8

```

##### ***cover_config.res***
1. 參數 pattern 為允許翻譯的檔案

```
[mapping]
pattern = *.txt;*.php;*.htm;*.html;*.tpl;*.tmpl;*.js

```





