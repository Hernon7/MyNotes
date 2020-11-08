# Guide to Making Git Contribution



## Create/Clone the repository

```
git init
```

```
git clone "url you just copied"
```

For example:

```
git clone https://github.com/this-is-you/first-contributions.git
```

## Git File Status

![](https://git-scm.com/book/en/v2/images/lifecycle.png)

## Edit Git Commend Alias

```

```

## Create a branch

Change to the repository directory on your computer (if you are not already there):

```
cd first-contributions
```

Now create a branch using the `git checkout` command:

```
git checkout -b <add-your-name>
```

For example:

```
git checkout -b add-alonzo-church
```

(The name of the branch does not need to have the word *add* in it, but it’s a reasonable thing to include because the purpose of this branch is to add your name to a list.)



## Make necessary changes and commit those changes

Now open `Contributors.md` file in a text editor, add your name to it, and then save the file. If you go to the project directory and execute the command `git status`, you'll see there are changes. Add those changes to the branch you just created using the `git add` command:

```
git add Contributors.md
```

Now commit those changes using the `git commit` command:

```
git commit -m "Add <your-name> to Contributors list"
```

replacing `<your-name>` with your name.

## Reset the change

```
git reset <commit id> -- hard or -- soft or -- mixed
git pull
```

## Check log

```
git log
git reflog
```

## Push changes

Push your changes using the command `git push`:

```
git push origin <add-your-name>
```

replacing `<add-your-name>` with the name of the branch you created earlier.

## Create new branch

```
git checkout -b <name> <template>
```

## Back to master

```
git checkout master
```

## Check all branch

```
git branch
```

## Merge

```
git merge <branchname>
```



## Keeping your fork synced with this repository

First, switch to the master branch.

```
git checkout master
```

Then add my repo’s url as `upstream remote url`:

```
git remote add upstream https://github.com/Roshanjossey/first-contributions
```

This is a way of telling git that another version of this project exists in the specified url and we’re calling it `upstream`. Once the changes are merged, fetch the new version of my repository:

```
git fetch upstream
```

Here we’re fetching all the changes in my fork (upstream remote). Now, you need to merge the new revision of my repository into your master branch.

```
git rebase upstream/master
```

Here you’re applying all the changes you fetched to master branch. If you push the master branch now, your fork will also have the changes:

```
git push origin master
```

Notice here you’re pushing to the remote named origin.

At this point I have merged your branch `<add-your-name>` into my master branch, and you have merged my master branch into your own master branch. Your branch is now no longer needed, so you may delete it:

```
git branch -d <add-your-name>
```

and you can delete the version of it in the remote repository, too:

```
git push origin --delete <add-your-name>
```

This isn’t necessary, but the name of this branch shows its rather special purpose. Its life can be made correspondingly short.