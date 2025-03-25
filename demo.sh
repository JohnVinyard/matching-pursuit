echo "Running Google Chrome in reduced-security mode so AudioWorklets can be loaded"

google-chrome \
    --args \
    --allow-file-access-from-files \
    --allow-file-access \
    --allow-cross-origin-auth-prompt \
    --new-window \
    "${1}"