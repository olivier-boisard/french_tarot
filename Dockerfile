FROM horizon:dev

ARG USERNAME

RUN groupadd $USERNAME
RUN useradd --create-home --gid $USERNAME $USERNAME
RUN chown $USERNAME:$USERNAME /home/miniconda/lib/python3.7/site-packages/
USER $USERNAME
