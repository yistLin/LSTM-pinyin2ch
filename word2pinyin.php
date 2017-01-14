<?php
require __DIR__ . '/vendor/autoload.php';
use Overtrue\Pinyin\Pinyin;

$pinyin = new Pinyin();

$file = fopen($argv[1], 'r');

while (!feof($file)) {
    $line = trim(fgets($file));
    $arr = $pinyin->convert($line);
    echo $line . "\t" . implode(' ', $arr) . "\n";
}

// echo implode(' ', $pinyin->convert('带着希望去旅行，比到达终点更美好')) . "\n";

fclose($file);
?>