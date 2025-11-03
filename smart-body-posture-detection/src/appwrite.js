import { Client, Account } from "appwrite";

const client = new Client();
client
    .setEndpoint("https://cloud.appwrite.io/v1") // or your Appwrite instance
    .setProject("6903c77b0020b7eeedc5");

export const account = new Account(client);
