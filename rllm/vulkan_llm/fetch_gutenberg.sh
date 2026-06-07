#!/bin/bash
# Fetch public domain books from Project Gutenberg
# Usage: bash fetch_gutenberg.sh [count]

COUNT=${1:-100}
DIR="GutenbergBooks"
mkdir -p "$DIR" "$DIR/tmp"

# Popular public domain book IDs
IDS=(
  84 11 1342 1661 98 74 2701 1260 76 2600
  345 1400 120 2554 2641 174 30601 4300 20796 2147
  17493 2199 1237 421 3207 1232 37106 27827 2814 26373
  305 730 25525 2852 521 24928 2383 996 16282 16
  221 2817 1934 6583 21076 7749 61250 1322 135 47183
  47692 3600 34901 8331 37 2130 28054 30368 34603 23334
  26654 44848 46852 17211 45943 59630 63857 2975 10155 54950
  590 4650 3614 22944 432 12336 2009 14591 545 58736
  9023 25824 54248 553 4045 16823 59184 8718 2711 316
  513 53506 33373 55681 8604 15210 20387 68847 9268 3162
)

for id in "${IDS[@]}"; do
  URL="https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
  OUTFILE="$DIR/tmp/pg$id.txt"
  if [ -f "$OUTFILE" ]; then
    echo "Already have $id, skipping"
    continue
  fi
  echo "Fetching book $id..."
  curl -sL --max-time 10 "$URL" -o "$OUTFILE"
  sleep 0.5
done

echo "Converting to .trdata..."
for f in "$DIR/tmp"/*.txt; do
  base=$(basename "$f" .txt)
  # Remove Gutenberg boilerplate (remove lines before/after the standard markers)
  out="$DIR/$base.trdata"
  awk '
    /^(\*\*\* START OF THE PROJECT GUTENBERG|^(\*\*\* |\*\*\*\*) START OF THIS PROJECT GUTENBERG)/ {p=1; next}
    /^(\*\*\* END OF THE PROJECT GUTENBERG|^(\*\*\* |\*\*\*\*) END OF THIS PROJECT GUTENBERG)/ {p=0}
    p
  ' "$f" > "$out"
done

total=$(cat "$DIR"/*.trdata 2>/dev/null | wc -c)
echo "Done. $total chars in $DIR/"
echo "Copy to TrainingData/:  cp $DIR/*.trdata TrainingData/"