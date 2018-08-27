# -*- coding: utf-8 -*-
import json
import logging

import requests

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def _update_caipirinha(base_url, item_path, token, item_id, data):
    headers = {'X-Auth-Token': token}

    if item_id == '':
        url = '{}/{}'.format(base_url, item_path)
    else:
        url = '{}/{}/{}'.format(base_url, item_path, item_id)

    log.debug(_('Querying Caipirinha URL: %s'), url)

    r = requests.post(url, headers=headers, data=data)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        raise RuntimeError(_(
            u"Error in URL {}: HTTP {} - {}").format(
            item_id, url, r.status_code, r.text))


def _emit_saving_visualization(emit_event_fn, task_id):  # pragma: no cover
    if emit_event_fn is not None:
        emit_event_fn(
            'update task', status='RUNNING',
            identifier=task_id,
            task={'id': task_id},
            message=_('Saving visualizations'),
            type='STATUS')


def _emit_saved_visualization(_type, emit_event_fn,
                              visualization):  # pragma: no cover
    if emit_event_fn is not None:
        emit_event_fn(
            'task result', status='COMPLETED',
            identifier=visualization['task_id'],
            task={'id': visualization['task_id']},
            message=_('Result generated'),
            type=_type,
            title=visualization['model'].title,
            operation={'id': visualization['model'].type_id},
            operation_id=visualization['model'].type_id)


def _emit_completed(emit_event_fn, task_id):  # pragma: no cover
    if emit_event_fn is not None:
        emit_event_fn(
            'update task', status='COMPLETED',
            identifier=task_id,
            task={'id': task_id},
            message=_('Visualizations saved'),
            type='STATUS')


def new_dashboard(config, title, user, workflow_id, workflow_name, job_id,
                  task_id, visualizations, emit_event_fn=None,
                  _type='VISUALIZATION'):
    data = dict(title=title, user=user, workflow_id=workflow_id,
                workflow_name=workflow_name, job_id=job_id, task_id=task_id,
                visualizations=visualizations)

    caipirinha_config = config['juicer']['services']['caipirinha']
    _emit_saving_visualization(emit_event_fn, task_id)

    for visualization in visualizations:
        _emit_saved_visualization(_type, emit_event_fn, visualization)
        del visualization['model']

    r = _update_caipirinha(caipirinha_config['url'], 'dashboards',
                           caipirinha_config['auth_token'], '',
                           json.dumps(data))
    _emit_completed(emit_event_fn, task_id)
    return r


def new_visualization(config, user, workflow_id, job_id,
                      task_id, visualization, emit_event_fn=None,
                      _type='VISUALIZATION'):
    caipirinha_config = config['juicer']['services']['caipirinha']

    _emit_saving_visualization(emit_event_fn, task_id)
    _emit_saved_visualization(_type, emit_event_fn, visualization)

    del visualization['model']
    r = _update_caipirinha(
        caipirinha_config['url'], 'visualizations',
        caipirinha_config['auth_token'], '', json.dumps(visualization))

    _emit_completed(emit_event_fn, task_id)
    return r
