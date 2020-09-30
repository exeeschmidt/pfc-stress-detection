import click
import os
from click.testing import CliRunner
import CommandLineInterface as Cli


@click.command(help="Detección de estrés")
@click.option(
    '-v', '--video',
    type=click.Path(file_okay=True, dir_okay=True, readable=True, resolve_path=True),
    default=None,
    help="Path del archivo de video"
)
@click.option(
    '-a', '--audio',
    type=click.Path(file_okay=True, dir_okay=True, readable=True, resolve_path=True),
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
    if video is None:
        return click.secho("Debe ingresar un archivo de video. Agregue --help para obtener más detalles.", fg='yellow')
    else:
        if not os.path.exists(video):
            return click.secho("ERROR: La ruta del video no existe, asegúrese de que sea una ruta válida.", fg='red')
        else:
            video_name = fileVeritication(str(video), [".mp4", ".avi"], "video")
            if video_name.find("ERROR") != -1:
                return click.secho(video_name, err=True, fg='red')
            else:
                click.secho("OK: Video " + click.format_filename(video_name), fg='green')

    audio_name = None
    if audio is not None:
        if not os.path.exists(audio):
            return click.secho("ERROR: La ruta del audio no existe, asegúrese de que sea una ruta válida.", fg='red')
        else:
            audio_name = fileVeritication(str(audio), ".wav", "audio")
            if audio_name.find("ERROR") != -1:
                return click.secho(audio_name, err=True, fg='red')
            else:
                click.secho("OK: Audio " + click.format_filename(audio_name), fg='green')

    # Si todo va bien, comienza a procesar
    if binarizar:
        click.secho("Salida expresada en estresado/no estrado", fg='green')
    else:
        click.secho("Salida expresada en niveles neutral/estrés bajo/estrés medio/estrés alto", fg='green')

    Cli.fileProcessing(video, video_name, audio, audio_name, binarizar)
    os._exit(0)
    return 0


def fileVeritication(file, extensions, arg):
    extension_output = ""
    for extension in extensions:
        extension_output += extension + " "
        if file.find(extension) != -1:
            idx1 = file.rfind(os.sep)
            idx2 = file.rfind('.')
            filename = file[idx1+1:idx2]
            return filename

    return "ERROR: El argumento \"--" + arg + "\" debe tener extensión " + extension_output + "."


# def test_cli():
#     runner = CliRunner()
#     result = runner.invoke(cli, ['--video', os.path.join("..", "Input", "Mauricio Macri 6seg.mp4")])
#     assert result.exit_code == 0
#     assert 'Debug mode is on' in result.output


if __name__ == '__main__':
    cli()
