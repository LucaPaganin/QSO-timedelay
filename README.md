# QSO-timedelay
Repository for time delay estimation of quasar light curves


## Important
Add these lines to the file `.git/config` of your local repository`
```
[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

This is to assure that each jupyter notebook you'll commit will automatically cleaned from its output.
