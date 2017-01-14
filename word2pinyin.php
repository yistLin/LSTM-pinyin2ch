<?php
require __DIR__ . '/vendor/autoload.php';
use Overtrue\Pinyin\Pinyin;

$pinyin = new Pinyin();

$file = fopen($argv[1], 'r');

while (!feof($file)) {
    $line = trim(fgets($file));
    $pline = $pinyin->sentence($line);
    echo $line . "\t" . $pline . "\n";
}

// echo implode(' ', $pinyin->convert('带着希望去旅行，比到达终点更美好 2 0 世紀')) . "\n";
// echo $pinyin->sentence('带着希望去旅行，比到达终点更美好 2 0 世紀') . "\n";

fclose($file);
?>