#!/bin/bash

# Build the documentation
cd docs
make html
cd ..

# Checkout to the gh-pages branch
git checkout gh-pages

# Remove all files
git rm -rf .

# Copy the contents of _build/html to the root directory
cp -r docs/_build/html/* .

# Add all files to git
git add .

# Commit the changes
git commit -m "Update documentation"

# Push the changes
git push origin gh-pages

# Checkout to the main branch
git checkout main
