# Contributing

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions.

## Table of Contents

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Your First Code Contribution](#your-first-code-contribution)
- [Improving The Documentation](#improving-the-documentation)
- [Style Guidelines](#style-guidelines)

## I Have a Question

Before you ask a question, search for existing [Issues](https://github.com/bjmorgan/site-analysis/issues) that might help you. If you find a suitable issue and still need clarification, you can write your question in that issue.

If you still need to ask a question:

- Open an [Issue](https://github.com/bjmorgan/site-analysis/issues/new)
- Provide as much context as you can about what you're running into
- Provide project and platform versions, depending on what seems relevant

We will then take care of the issue as soon as possible.

## I Want To Contribute

### Reporting Bugs

#### Before Submitting a Bug Report

Please complete the following steps to help us fix any potential bug as fast as possible:

- Make sure that you are using the latest version
- Check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/bjmorgan/site-analysis/issues?q=label%3Abug)
- Collect information about the bug:
  - Stack trace (if there's an error)
  - Python version and package versions
  - Steps to reproduce the issue
  - Expected vs actual behaviour

#### How Do I Submit a Good Bug Report?

- Open an [Issue](https://github.com/bjmorgan/site-analysis/issues/new)
- Explain the behaviour you would expect and the actual behaviour
- Provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own
- Provide the information you collected in the previous section

Once it's filed:

- The project team will label the issue accordingly
- A team member will try to reproduce the issue with your provided steps
- If the team is able to reproduce the issue, it will be available for [someone to implement a fix](#your-first-code-contribution)

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, **including completely new features and minor improvements to existing functionality**.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version
- Read the [documentation](https://site-analysis.readthedocs.io) carefully and find out if the functionality is already covered
- Perform a [search](https://github.com/bjmorgan/site-analysis/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one
- Find out whether your idea fits with the scope and aims of the project

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/bjmorgan/site-analysis/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible
- **Describe the current behaviour** and **explain which behaviour you expected to see instead** and why
- **Explain why this enhancement would be useful** to most users

### Your First Code Contribution

1. [Open an issue](https://github.com/bjmorgan/site-analysis/issues/new) to discuss what you'd like to change or fix
2. Once you've discussed your proposed changes in an issue:
   - Fork the repository
   - Create a new branch for your changes
   - Make your changes
   - Submit a pull request that references the issue (e.g., "Fixes #123")

**Requirements for all contributions:**

**Unit Tests**: All new functionality must include unit tests. Ensure your tests:
- Cover the main functionality you're adding
- Include edge cases where appropriate
- Pass before submitting your pull request

We encourage Test-Driven Development (TDD) - ideally provide a failing test before providing a fix (although this is non-essential).

Run tests with:
```bash
pytest
```

**Documentation**: Use Google-style docstrings for all functions and classes:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of what the function does.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.
        
    Returns:
        Description of what the function returns.
        
    Raises:
        ValueError: When param2 is negative.
    """
```

### Improving The Documentation

Documentation improvements follow the same process as code contributions:

1. [Open an issue](https://github.com/bjmorgan/site-analysis/issues/new) to discuss the documentation changes you'd like to make
2. Create a pull request that references the issue (e.g., "Fixes #123")

## Style Guidelines
- The project uses British English in documentation and comments
- **Exception**: Use American spelling "center" (not "centre") in all API parameter names, method names, and public interfaces for consistency with the Python scientific ecosystem (NumPy, SciPy, PyMatGen)
