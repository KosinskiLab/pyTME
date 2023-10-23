.. include:: ../substitutions.rst

====================
Contributor's Guide
====================

We greatly appreciate community contributions to our project. If you're thinking of adding new features, fixing bugs, or making any changes, this guide is designed to get you started and ensure the consistency and quality of the codebase.

Code Style and Guidelines
--------------------------

1. **Code Formatting**: We follow PEP 8, the style guide for Python code. We recommend using tools like `black` to check your code for PEP 8 compliance.

2. **Docstrings**: All functions, classes, and methods should have numpydoc-compliant docstrings. This ensures that our documentation remains consistent and readable. See the `numpydoc` guide for reference.

3. **No Unnecessary Getters and Setters**: Avoid using getters and setters unless they provide clear value, such as data validation or computed properties. Direct attribute access is more Pythonic.

4. **Minimize Line Count**: While it's important to write clear code, also aim for brevity. Don't use five lines of code when one will do. But remember, clarity should not be sacrificed for brevity.

5. **Commit Messages**: Write meaningful commit messages. A good rule of thumb is to make the commit message complete the sentence "If applied, this commit will..."

6. **Testing**: Write tests for any new features or bug fixes. Ensure that all tests pass before submitting a pull request.

7. **Stay Updated**: Make sure your fork and local copies are updated regularly from the main branch to avoid merge conflicts.

Contribution Workflow
----------------------

1. **Fork the Repository**: If you're new to the project, first create a fork of the main repository.

2. **Clone Your Fork Locally**:

   .. code-block:: bash

      $ git clone https://github.com/KosinskiLab/pyTME.git

3. **Set Upstream Remote**: This allows you to pull in changes from the main repository.

   .. code-block:: bash

      $ git remote add upstream https://github.com/KosinskiLab/pyTME.git

4. **Make and Commit Your Changes**: Remember to keep your changes focused on one thing, whether it's a new feature or a bug fix.

5. **Pull the Latest Changes**: Before pushing your changes, pull the latest updates from the master repository.

   .. code-block:: bash

      $ git pull upstream master

6. **Push to Your Fork and Create a Pull Request**: After resolving any potential conflicts, push your changes to your fork and create a pull request against the master repository.

Feedback and Review
-------------------

Once you've submitted your pull request:

1. **Maintainer Review**: One of the maintainers will review your pull request. They may provide feedback or request changes. This is a collaborative process and an opportunity to ensure the quality and consistency of the codebase.

2. **Continuous Integration**: Ensure that your pull request passes all CI checks and tests.

3. **Merging**: Once approved, your pull request will be merged into the main branch.

Thank you for considering contributing to our project! Your time and effort help improve |project| for everyone.

