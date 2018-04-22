# deeplearning
Deeplearning Udacity

# E2I
Para descargar, ejectuar desde el terminal:
git clone https://github.com/Eduzc07/deeplearning.git


## Create a branch
Create the branch on your local machine and switch in this branch :
```
git checkout -b [name_of_your_new_branch]
```
Change working branch :
```
 git checkout [name_of_your_new_branch]
```
Push the branch on github :
```
 git push origin [name_of_your_new_branch]
```
You can see all branches created by using :
```
git branch
git branch -r
git branch -a
```
:warning: Only Admin!!\
Merge with master:
```
 git checkout master
 git merge [name_of_your_new_branch]
```
Merge master into Branch:
```
 git checkout {branch}
 git merge master
```

Remove Local branch:
```
 git branch -d {the_local_branch}
```
Remove a remote branch:
```
 git push origin --delete {the_remote_branch}
```

Permanently authenticating with Git repositories:
```
 git config credential.helper store
 git push https://github.com/repo.git
 Username for 'https://github.com': <USERNAME> -- > Eduzc07
 Password for 'https://USERNAME@github.com': <PASSWORD>
```

# Run Always from Branch
```
git add {files}
git commit -m "Description"
git push origin {branch_name}
```

## Info
[Create a new branch](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches)\
[GIt Branching](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)\
[E2 Innovation S.R.L.](http://www.e2i.com.pe/)

