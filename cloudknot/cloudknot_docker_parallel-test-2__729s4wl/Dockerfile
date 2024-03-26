###############################################################################
# Dockerfile to build parallel-test-2 application container
# Based on python:3.11
###############################################################################

# Use official python base image
FROM python:3.11

# Install python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir git+https://github.com/asagilmore/dipy.git@parallel_test

# Create a default user. Available via runtime flag `--user cloudknot-user`.
# Add user to "staff" group.
# Give user a home directory.
RUN groupadd -f staff \
    && useradd --create-home --groups staff "cloudknot-user"

# Copy the python script
COPY --chown="cloudknot-user" "parallel-test-2.py" "/home/cloudknot-user/"

# Make the python script executable
RUN chmod +x "/home/cloudknot-user/parallel-test-2.py"

# Set user
USER "cloudknot-user"

# Set working directory
WORKDIR "/home/cloudknot-user"

# Set entrypoint
ENTRYPOINT ["/home/cloudknot-user/parallel-test-2.py"]
