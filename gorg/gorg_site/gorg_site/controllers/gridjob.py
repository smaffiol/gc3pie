import logging

from pylons import request, response, session, tmpl_context as c
from pylons.controllers.util import abort, redirect_to
from gorg_site.lib.base import BaseController, render
from pylons.decorators import jsonify
import os
from gorg_site.model.gridjob import GridjobModel
from gorg_site.lib.mydb import Mydb

log = logging.getLogger(__name__)
PERMANENT_STORE = '/home/mmonroe/uploads/'

class GridjobController(BaseController):

    def index(self):
        # Return a rendered template
        #return render('/gridjob.mako')
        # or, return a response
        return 'Hello World'

    @jsonify
    def json_test(self):
        if request.environ['CONTENT_TYPE'] == 'application/json':
            return {'response':'I am json'}
        return render('/submit_job_form.mako')
        
    def submit_form(self):
        return render('/submit_job_form.mako')
    
    def create(self):
        """Post / users: Create a new job in the database."""
        # Myfile is a fieldstorage object provided by pylons
        # myfile.filename = basename of file, myfile.file = file like object
        # myfile.value = contents of file
        myfile = request.POST['myfile'] 
        title = request.POST['title']
        author = request.POST['author']
        db=Mydb().cdb()
        a_job = GridjobModel()
        a_job.author = author
        a_job.title = title
        a_job.defined_type = 'GAMESS'
        a_job.put_attachment(db, myfile.file, myfile.filename)
        c.mess = 'Successfully uploaded: %s, title: %s' % \
                (myfile.filename, title)
        return render('/submit_job_finish.mako')

    def query_job(self):
        """Post / users: Query an existing job in the database."""
        jobid = request.GET['jobid'] 
        db=Mydb().cdb()
        a_job = GridjobModel().load(db,jobid)
        c.a_job = a_job
        return render('/submit_job_detail.mako')
        
    def display_job_attachment(self):
        """Post / users: Query an existing job in the database."""
        jobid = request.GET['jobid'] 
        attachment = request.GET['attachment']
        db=Mydb().cdb()
        a_job = GridjobModel().load(db,jobid)
        response.content_type = 'text/plain'
        return a_job.get_attachment(db, attachment)
   

