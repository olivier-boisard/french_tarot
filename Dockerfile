FROM horizon:dev

ARG USERNAME
ARG USERID

RUN useradd --create-home --user-group --uid $USERID $USERNAME
RUN chown $USERNAME:$USERNAME /home/miniconda/lib/python3.7/site-packages/
USER $USERNAME
