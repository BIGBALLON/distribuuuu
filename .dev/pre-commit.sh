# Run this script at project root 
# by "bash ./dev/pre-commit.sh" before you commit.

{
	black --version | grep "20.8b1" > /dev/null
} || {
    echo `black --version`
	echo "requires black==20.8b1 !"
	exit 1
}

echo "Running isort..."
isort --profile black .

echo "Running black..."
black .

echo "Running flake8..."
flake8 . --config ./setup.cfg

echo "done."

