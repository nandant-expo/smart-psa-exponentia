import { Injectable } from '@angular/core';
import { BlobServiceClient, ContainerClient } from '@azure/storage-blob';
import { BehaviorSubject } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class BlobStorageService {
  
  sas = environment.sas;
  containerName = environment.containerName;
  
  constructor() { }

  async listFiles(): Promise<String[]> {
    let result: String[] = [];
    let blobs = this.containerClient().listBlobsFlat();
    for await (const blob of blobs) {
      result.push(blob.name as String)
    }
    return result;
  }

  containerClient(): ContainerClient {
    return new BlobServiceClient(environment.bloburl+this.sas)
      .getContainerClient(this.containerName);
  }

}
