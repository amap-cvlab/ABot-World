# ABot-World Project Page

Publish `website/` as the root of the static site. In the repository settings,
open **Pages** and set **Build and deployment → Source** to **GitHub Actions**.
The **Deploy Project Page** workflow uploads this directory and serves
`website/index.html` at the Pages root.

Run this command from the repository root to preview the site locally.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory website
```
