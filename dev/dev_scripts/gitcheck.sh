git status -u --porcelain | cut -c4- | xargs du -h | sort -rh | awk '{printf "%-10s %s\n", $1, $2}'
