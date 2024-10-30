import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import { BehaviorSubject } from 'rxjs';

@Injectable()
export class ChatService {
  querySent: BehaviorSubject<boolean> = new BehaviorSubject<boolean>(false);
  query: BehaviorSubject<string> = new BehaviorSubject<string>('');
  chatFlow: BehaviorSubject<string> = new BehaviorSubject<string>('');
  newQuery: BehaviorSubject<any> = new BehaviorSubject<any>(null);
  stream: BehaviorSubject<boolean> = new BehaviorSubject<boolean>(false);
  constructor(private http: HttpClient) { }

  getChatMessageHistoryList(formData: any) {
    return this.http.post(environment.apiendpoint + 'table_chat_history_psa', formData , { responseType: 'json' });
  }

  getChatMessages(formData: any) {
    return this.http.post(environment.apiendpoint + 'table_chats_psa', formData , { responseType: 'json' });
  }

  getChatMessage(formData: any) {
    return this.http.post(environment.apiendpoint + 'table_get_chat', formData , { responseType: 'json' });
  }
  
  deleteChatMessageHistory(formData: any) {
    return this.http.post(environment.apiendpoint + 'table_chat_delete_psa', formData , { responseType: 'json' });
  }

  getSDAType(formData: any){
    return this.http.post(environment.apiendpoint + 'decision_making_psa', formData , { responseType: 'json' });
  }

  getSDAVisualization(formData: any){
    return this.http.post(environment.apiendpoint + 'visualisation_psa', formData , { responseType: 'json' });
  }

  chatExistingFileProcess(formData: any) {
    return this.http.post(environment.apiendpoint + 'table_process', formData , { responseType: 'json' });
  }

  saveFeedBack(formData: any){
    return this.http.post(environment.apiendpoint + 'table_feedback_toggle_psa', formData , { responseType: 'json' });
  }

  //common functions

  dragElement(element: any, direction: any){
    var   md: any; // remember mouse down info
    const first  = document.getElementById("first") as HTMLElement;
    const second = document.getElementById("second") as HTMLElement;
    first.style.width = "70%";
    second.style.width = "30%";
    element.onmousedown = onMouseDown;

    function onMouseDown(e: any){
        md = {e,
              offsetLeft:  element.offsetLeft,
              offsetTop:   element.offsetTop,
              firstWidth:  first.offsetWidth,
              secondWidth: second.offsetWidth
            };

        document.onmousemove = onMouseMove;
        document.onmouseup = () => {
            document.onmousemove = document.onmouseup = null;
        }
    }

    function onMouseMove(e: any){
        var delta = {x: e.clientX - md.e.clientX,
                    y: e.clientY - md.e.clientY};

        if (direction === "H" ) // Horizontal
        {
            // Prevent negative-sized elements
            delta.x = Math.min(Math.max(delta.x, -md.firstWidth),
                      md.secondWidth);

            element.style.left = md.offsetLeft + delta.x + "px";
            let firstNewWidth = (md.firstWidth + delta.x) // console.log(md.secondWidth - delta.x)
            let secondNewWidth = (md.secondWidth - delta.x) // console.log(md.secondWidth - delta.x)

            if((firstNewWidth/window.innerWidth) * 100 <= 22){
              first.style.width = "22%";
              second.style.width = "78%";
            }else if((secondNewWidth/window.innerWidth) * 100 <= 22){
              first.style.width = "78%";
              second.style.width = "22%";
            }
            else{
              first.style.width = firstNewWidth + "px";
              second.style.width = secondNewWidth + "px";
            }
        }
    }
  }

  exportAsTxt(txtData: any,filename: any){
    var data = new Blob([txtData], {type: 'text/plain;charset=utf-8'});

    let url = window.URL.createObjectURL(data);

    let a = document.createElement('a');
    document.body.appendChild(a);

    a.setAttribute('style', 'display: none');
    a.href = url;
    a.download = "AIXPONENT-"+  filename+'.txt';
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  }
}