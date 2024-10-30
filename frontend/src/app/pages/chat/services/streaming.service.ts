
import { Injectable } from '@angular/core';
import { environment } from '../../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class StreamingService {
  constructor() { }
  
  async chatQueryStreaming(formData: any){
    try {
      let token = localStorage.getItem("access_token");
      const response = await fetch(environment.apiendpoint + 'stream_chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + token
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.body?.getReader();
    } catch (error: any) {
      // Handle errors
      console.error('Error:', error.message);
      return false;
    }
  }
}
