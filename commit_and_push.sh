#!/bin/bash
REPO_NAME=$NGS
git init
git add .
git commit -m "automated commit"
git remote add origin https://github.com/iharuto/${REPO_NAME}.git
git push -u origin main
