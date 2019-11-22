FROM horizon:dev

ARG USERID
ARG USERGROUP
ARG WORKDIR

RUN groupadd $USERGROUP
RUN useradd -g $USERGROUP $USERID
RUN chown $USERID:$USERGROUP /home/miniconda/lib/python3.7/site-packages/
USER $USERID
