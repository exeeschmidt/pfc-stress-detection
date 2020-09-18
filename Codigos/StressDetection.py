import click
import os
import main

@click.command(help="Detección de estrés")
@click.option(
    '-v', '--video',
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
    default=None,
    help="Path del archivo de video"
)
@click.option(
    '-a', '--audio',
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
    default=None,
    help="Path del archivo de audio"
)
@click.option(
    '-b', '--binarizar',
    is_flag=True,
    type=click.BOOL,
    default=False,
    help="Obtener salidas binarias o niveles de estrés"
)
def cli(video, audio, binarizar):
    if video is None and audio is None:
        click.secho("Ingrese --help para obtener más detalles", fg='yellow')

    if video is not None:
        path_video = str(video)
        if videoProcessing(path_video):
            main.main()

    if audio is not None:
        path_audio = str(audio)
        audioProcessing(path_audio)

    if binarizar:
        click.secho("Salida binaria", fg='green')
    else:
        click.secho("Salida en niveles de estrés", fg='green')


def videoProcessing(path):
    if path.find(".mp4") != -1:
        idx = path.rfind(os.sep)
        if idx == -1:
            click.secho("Procesando " + click.format_filename(path), fg='green')
        else:
            click.secho("Procesando " + click.format_filename(path[idx+1:]), fg='green')
    else:
        click.secho("ERROR: El argumento \"--video\" debe tener extensión .mp4.", fg='red')
        return False
    return True


def audioProcessing(path):
    if path.find(".mp3") != -1:
        idx = path.rfind(os.sep)
        if idx == -1:
            click.secho("Procesando " + click.format_filename(path), fg='green')
        else:
            click.secho("Procesando " + click.format_filename(path[idx+1:]), fg='green')
    else:
        click.secho("ERROR: El argumento \"--audio\" debe tener extensión .mp3.", fg='red')


if __name__ == '__main__':
    cli()
