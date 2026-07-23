# Веса модели

Файл `model_alzheimer_3dcnn.pth` (~127 МБ) уже лежит в этой папке, но **не закоммичен
в git** — в окружении, где собирался репозиторий, не было доступно `git-lfs`.

GitHub не принимает файлы тяжелее 100 МБ как обычные blob'ы, поэтому перед пушем
нужно один раз настроить Git LFS локально:

```bash
git lfs install
git lfs track "*.pth"       # уже прописано в .gitattributes, но на всякий случай
git add .gitattributes
git add models/model_alzheimer_3dcnn.pth
git commit -m "Add trained model weights via Git LFS"
git push
```

Если `git-lfs` не установлен:

- macOS: `brew install git-lfs`
- Ubuntu/Debian: `sudo apt install git-lfs`
- Windows: `winget install GitHub.GitLFS` (или инсталлятор с https://git-lfs.com)

Альтернатива, если не хочется использовать LFS: хранить веса вне git (например,
в Kaggle Models / Hugging Face Hub / облачном хранилище) и подтягивать их отдельным
скриптом при развёртывании, оставив в репозитории только ссылку на них.
