GADIN: Generative Adversarial Denoise Imputation Network for Incomplete Data
## Dependency issue fixed 
 - fix dependencies issue
 - python version used in project: 3.10
 - if problem appears when trying to install requeirements.txt, try to manually install with command `pip install <package name>`

## Run project 

```bash
$ # Download
$ git clone https://github.com/yaozeliang/Library.git
$ cd Library

$ # Install vierual environment
$ pip install virtualenv
$ virtualenv envname

$ # Activae virtual env
$ envname\scripts\activate

$cd ../..
$ pip3 install -r requirements.txt
$
$ # Create tables
$ python manage.py makemigrations
$ python manage.py migrate
$
$ # Start the application 
$ python manage.py runserver 

```

