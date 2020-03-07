FROM akashin/coursera-aml-nlp:latest

RUN apt update && \
    apt install -y \
        openssh-server

RUN mkdir /var/run/sshd
RUN sed -ri 's/^#?UMASK\s+.*/UMASK 002/' /etc/login.defs && \
    sed -ri "s/\/usr\/lib\/openssh\/sftp-server/internal-sftp/g" /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]